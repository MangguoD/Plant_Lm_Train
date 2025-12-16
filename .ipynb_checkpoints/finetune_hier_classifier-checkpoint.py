#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分段形式（5000 × 64）下游表型分类微调脚本（单机单卡）

核心设计：
    - 每个样本包含 num_segments=5000 个片段，每片段长度 seg_len=64（单位：6-mer token）
    - 先用预训练 DNAModel 对每个片段做编码： [B, 5000, 64] → [B, 5000, 64, d_model]
    - 在片段内部做 mean pooling：       → [B, 5000, d_model]
    - 对 5000 个片段再 pooling：         → [B, d_model]
    - 接线性分类头，做交叉熵训练

用法（对一个任务）示例：
    python finetune_hier_classifier.py \
        --ckpt dna_model/model_best.pt \
        --task_file ./downstream/data/zhugao.txt \
        --out_path ./downstream/model/zhugao_classifier.pth \
        --data_dir ./data \
        --device auto
"""

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from plant_lm_train import Config, DNAModel  # 你的预训练代码里的定义


# ====================
# 与预训练一致的工具函数
# ====================

def six_mer_to_index(six_mer: str) -> int:
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    idx = 0
    for i, b in enumerate(six_mer):
        idx += base_map[b] * (4 ** (5 - i))
    return idx


def clean_sequence(seq: str) -> str:
    return seq.upper().replace("N", "")


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] CUDA requested but not available, fallback to CPU.")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ====================
# 下游数据集：每样本 5000×64
# ====================

class SoySegmentDataset(Dataset):
    """
    输入文件可以是 txt/csv/tsv，只要：
        - 第一行是表头
        - 列之间用统一分隔符（tab / 空格 / 逗号等）

    列结构大致为：
        Sample_ID, Phenotype, geneType1, geneType2, ..., geneTypeK(≈5000)

    每个 geneTypeX 对应一个“片段”，长度为 seg_len=64（单位：6-mer token）。

    默认假设 geneTypeX 列里存的是 "ACGT..." 碱基字符串，
    会在这里再切分成 6-mer → bucket index；如果你那边已经是 index 序列，
    改 dna_segment_to_tokens() 即可。
    """

    def __init__(
        self,
        file_path: Path,
        config: Config,
        label_col: str = "Phenotype",
        gene_prefix: str = "geneType",
        seg_len: int = 64,
        max_segments: int | None = 5000,
    ):
        super().__init__()
        self.config = config
        self.label_col = label_col
        self.gene_prefix = gene_prefix
        self.seg_len = seg_len
        self.max_segments = max_segments

        # 自动推断分隔符（支持 txt/csv/tsv）
        self.df = pd.read_csv(file_path, sep=None, engine="python")

        # all geneType* columns
        gene_cols = [c for c in self.df.columns if c.startswith(self.gene_prefix)]
        if not gene_cols:
            raise ValueError(
                f"No columns starting with '{self.gene_prefix}' found in {file_path}"
            )

        def _suffix_int(c: str) -> int:
            s = c[len(self.gene_prefix):]
            return int(s) if s.isdigit() else 0

        gene_cols = sorted(gene_cols, key=_suffix_int)

        if self.max_segments is not None:
            gene_cols = gene_cols[: self.max_segments]

        self.gene_cols = gene_cols
        print(
            f"[INFO] Using {len(self.gene_cols)} geneType segments per sample "
            f"(from columns {self.gene_cols[0]} .. {self.gene_cols[-1]})"
        )

        # 标签映射
        raw_labels = self.df[self.label_col].tolist()
        uniq_labels = sorted(set(raw_labels))
        self.label_mapping = {v: i for i, v in enumerate(uniq_labels)}
        self.labels = [self.label_mapping[v] for v in raw_labels]

        print(
            f"[INFO] Loaded {len(self.df)} samples from {file_path}, "
            f"num_classes={len(self.label_mapping)}, label_map={self.label_mapping}"
        )

    def dna_segment_to_tokens(self, dna: str) -> list[int]:
        """
        把一个 geneType 片段转成长度 seg_len 的 6-mer bucket 序列。

        默认 dna 是碱基串；如果你那边存的是“空格分隔的整数列表”，
        这里改成解析 int，然后 pad/截断到 seg_len 即可。
        """
        dna = clean_sequence(str(dna))
        tokens = []
        for i in range(0, len(dna) - 5, 6):
            mer = dna[i:i + 6]
            if len(mer) == 6 and all(b in "ACGT" for b in mer):
                tokens.append(six_mer_to_index(mer))
            if len(tokens) >= self.seg_len:
                break

        if len(tokens) == 0:
            tokens = [0] * self.seg_len
        elif len(tokens) < self.seg_len:
            rep = (self.seg_len + len(tokens) - 1) // len(tokens)
            tokens = (tokens * rep)[: self.seg_len]
        elif len(tokens) > self.seg_len:
            tokens = tokens[: self.seg_len]

        return tokens

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = self.labels[idx]

        seg_tokens = []
        for col in self.gene_cols:
            dna_seg = row[col]
            tokens = self.dna_segment_to_tokens(dna_seg)
            seg_tokens.append(tokens)

        seg_tokens = torch.tensor(seg_tokens, dtype=torch.long)  # [N_seg, L]

        return {
            "tokens": seg_tokens,                  # [N_seg, L]
            "label": torch.tensor(label).long(),   # []
        }


def soy_collate_fn(batch):
    tokens_list = [b["tokens"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    tokens = torch.stack(tokens_list, dim=0).contiguous()  # [B, N_seg, L]
    B, N_seg, L = tokens.shape

    # 注意：expand 之后立刻 clone，打断共享 storage，避免 pin_memory 报错
    pos = (
        torch.arange(L, dtype=torch.long)
        .view(1, 1, L)
        .expand(B, N_seg, L)
        .clone()                # 关键一行
        .contiguous()
    )

    return {
        "tokens": tokens,
        "positions": pos,
        "label": labels,
    }


# ====================
# 分层分类模型：LM + pooling + 线性头
# ====================

class HierDNAClassifier(nn.Module):
    def __init__(self, lm: DNAModel, num_classes: int, freeze_lm: bool = True):
        super().__init__()
        self.lm = lm
        self.freeze_lm = freeze_lm

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False
            # 冻结时直接 eval，关掉 dropout 等
            self.lm.eval()

        d_model = self.lm.config.d_model
        self.classifier = nn.Linear(d_model, num_classes)

    def encode_segments(self, tokens, positions, species_ids):
        """
        tokens:   [B, N_seg, L]
        positions:[B, N_seg, L]
        species_ids: [B]
        """
        B, N_seg, L = tokens.shape
        device = next(self.lm.parameters()).device

        tokens = tokens.to(device)
        positions = positions.to(device)
        species_ids = species_ids.to(device)

        # 展平成 [B*N_seg, L]
        tokens_flat = tokens.view(B * N_seg, L)
        positions_flat = positions.view(B * N_seg, L)
        species_flat = species_ids.unsqueeze(1).expand(B, N_seg).reshape(B * N_seg)

        # 冻结 backbone 时，不保存梯度；否则正常反向
        ctx = torch.no_grad() if self.freeze_lm else torch.enable_grad()
        with ctx:
            token_embeds = self.lm.token_embed(tokens_flat)
            pos_embeds = self.lm.pos_embed(positions_flat)
            species_embeds = self.lm.species_embed(species_flat).unsqueeze(1)

            x = token_embeds + pos_embeds + species_embeds

            # 关键修改：下游任务里关闭 GlobalAttention，只用 LocalAttention + FFN
            bucket_matrix = None

            for layer in self.lm.layers:
                x = layer(x, positions_flat, bucket_matrix=bucket_matrix)

            x = self.lm.final_norm(x)      # [B*N_seg, L, D]

            seg_repr = x.mean(dim=1)       # [B*N_seg, D]  对每个 segment 内做 mean pooling

        seg_repr = seg_repr.view(B, N_seg, -1)  # [B, N_seg, D]
        return seg_repr

    def forward(self, tokens, positions, species_ids):
        seg_repr = self.encode_segments(tokens, positions, species_ids)
        # 对 5000 个片段再做一次 mean pooling
        sample_repr = seg_repr.mean(dim=1)  # [B, D]
        logits = self.classifier(sample_repr)
        return logits


# ====================
# 加载预训练 LM
# ====================

def load_pretrained_lm(ckpt_path: Path, data_dir: Path, device: torch.device):
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    epoch = ckpt.get("epoch", None)
    best_val_loss = ckpt.get("best_val_loss", None)
    if epoch is not None:
        print(f"[INFO] Checkpoint epoch        : {epoch}")
    if best_val_loss is not None:
        print(f"[INFO] Checkpoint best_val_loss: {best_val_loss:.4f}")

    if "config" in ckpt:
        config = ckpt["config"]
        print("[INFO] Loaded Config from checkpoint.")
    else:
        config = Config()
        print("[WARN] No 'config' in checkpoint, using fresh Config().")

    # 构建 species_list（与预训练一致）
    species = []
    if data_dir.exists() and data_dir.is_dir():
        for d in sorted(data_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                species.append(d.name)
    if not species:
        species = ["Glycine_max", "Glycine_soja"]
        print(f"[WARN] No species subdirs in {data_dir}, fallback to {species}")
    else:
        print(f"[INFO] Detected species_list from data: {species}")

    lm = DNAModel(config, species_list=species)
    state_dict = ckpt["model_state_dict"]
    missing_keys, unexpected_keys = lm.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(
            f"[WARN] Missing keys: {len(missing_keys)}, "
            f"Unexpected keys: {len(unexpected_keys)}"
        )
        if missing_keys:
            print("  Missing (first 10):", missing_keys[:10])
        if unexpected_keys:
            print("  Unexpected (first 10):", unexpected_keys[:10])
    else:
        print("[INFO] LM state_dict loaded cleanly.")

    lm.to(device)
    return lm, config, species


# ====================
# 训练循环
# ====================

def train_classifier(
    classifier: HierDNAClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
):
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = math.inf
    best_state = None

    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        total = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            positions = batch["positions"].to(device)
            labels = batch["label"].to(device)

            # 这里暂时统一用 0 号物种（假设都是 Glycine_max）
            species_ids = torch.zeros(tokens.size(0), dtype=torch.long, device=device)

            logits = classifier(tokens, positions, species_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tokens.size(0)
            total += tokens.size(0)

        avg_train_loss = running_loss / max(total, 1)

        # 验证
        classifier.eval()
        val_loss = 0.0
        val_total = 0
        correct = 0

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                positions = batch["positions"].to(device)
                labels = batch["label"].to(device)
                species_ids = torch.zeros(tokens.size(0), dtype=torch.long, device=device)

                logits = classifier(tokens, positions, species_ids)
                loss = criterion(logits, labels)

                val_loss += loss.item() * tokens.size(0)
                val_total += tokens.size(0)

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = correct / max(val_total, 1)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                "model_state_dict": classifier.state_dict(),
                "best_val_loss": best_val_loss,
            }

    return best_state


# ====================
# main
# ====================

def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical finetuning for phenotype classification")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to pretrained LM checkpoint, e.g. dna_model/model_best.pt")
    p.add_argument("--task_file", type=str, required=True,
                   help="Downstream txt/csv/tsv path (每行包含 ~5000 个 geneType 列)")
    p.add_argument("--data_dir", type=str, default="./data",
                   help="Species data dir used in pretraining (for species_list)")
    p.add_argument("--out_path", type=str, required=True,
                   help="Where to save finetuned classifier .pth")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size in units of samples（每个样本已经包含5000个片段）")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze_lm", action="store_true",
                   help="Freeze LM parameters (建议先开着)")
    p.add_argument("--no_freeze_lm", dest="freeze_lm", action="store_false")
    p.set_defaults(freeze_lm=True)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(42)
    random.seed(42)

    ckpt_path = Path(args.ckpt)
    task_file = Path(args.task_file)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path)
    device = select_device(args.device)

    print(f"[INFO] Using device : {device}")
    print(f"[INFO] Task file    : {task_file}")
    print(f"[INFO] Output path  : {out_path}")

    lm, config, species_list = load_pretrained_lm(ckpt_path, data_dir, device)

    full_ds = SoySegmentDataset(task_file, config=config)
    num_classes = len(full_ds.label_mapping)

    # 简单 8:2 划分 train/val
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=soy_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=soy_collate_fn,
    )

    classifier = HierDNAClassifier(lm, num_classes=num_classes, freeze_lm=args.freeze_lm)
    print(f"[INFO] freeze_lm = {args.freeze_lm}")
    print(f"[INFO] num_classes = {num_classes}, label_map = {full_ds.label_mapping}")

    best_state = train_classifier(
        classifier,
        train_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        lr=args.lr,
    )

    if best_state is None:
        print("[WARN] No best_state captured, nothing saved.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_state["config"] = config
    best_state["species_list"] = species_list
    best_state["label_mapping"] = full_ds.label_mapping
    best_state["freeze_lm"] = args.freeze_lm

    torch.save(best_state, out_path)
    print(f"[DONE] Saved classifier to {out_path}")


if __name__ == "__main__":
    main()