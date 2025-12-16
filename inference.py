#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对单条 DNA 序列做推理，输出其分类。

用法示例：
    python inference_hier_classifier.py \
        --model_path ./downstream/model/zhugao_classifier.pth \
        --sequence "ACGTACGT...." \
        --device auto
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from plant_lm_train import DNAModel, Config

# ===========================================================
# 与训练完全一致的工具函数
# ===========================================================

def six_mer_to_index(six_mer: str) -> int:
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    idx = 0
    for i, b in enumerate(six_mer):
        idx += base_map[b] * (4 ** (5 - i))
    return idx

def clean_sequence(seq: str) -> str:
    return seq.upper().replace("N", "")

# ===========================================================
# 单条序列 → 5000 × 64 token 片段
# ===========================================================

def sequence_to_segments(seq: str, num_segments=5000, seg_len=64):
    """
    将一条完整序列按训练时分段方式切成 [5000, 64] token 结构。
    若序列不够长，则循环复制；若太长，则截断。
    """
    seq = clean_sequence(seq)

    # 切成连续 6-mer
    mers = []
    for i in range(0, len(seq) - 5, 6):
        mer = seq[i:i+6]
        if len(mer) == 6 and all(x in "ACGT" for x in mer):
            mers.append(six_mer_to_index(mer))

    if len(mers) == 0:
        mers = [0]

    # 展平成一个足够长的 token 列表
    need = num_segments * seg_len
    rep = (need + len(mers) - 1) // len(mers)
    long_list = (mers * rep)[:need]

    # 组装成 [num_segments, seg_len]
    tokens = []
    for i in range(num_segments):
        seg = long_list[i * seg_len : (i + 1) * seg_len]
        tokens.append(seg)

    return torch.tensor(tokens, dtype=torch.long)   # [5000, 64]


# ===========================================================
# 加载微调模型（LM + 分类头）
# ===========================================================

class HierDNAClassifier(nn.Module):
    """与训练脚本一致"""
    def __init__(self, lm: DNAModel, num_classes: int, freeze_lm: bool = True):
        super().__init__()
        self.lm = lm
        self.freeze_lm = freeze_lm

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False
            self.lm.eval()

        d_model = self.lm.config.d_model
        self.classifier = nn.Linear(d_model, num_classes)

    def encode_segments(self, tokens, positions, species_ids):
        B, N_seg, L = tokens.shape
        device = next(self.lm.parameters()).device

        tokens = tokens.to(device)
        positions = positions.to(device)
        species_ids = species_ids.to(device)

        tokens_flat = tokens.view(B * N_seg, L)
        positions_flat = positions.view(B * N_seg, L)
        species_flat = species_ids.unsqueeze(1).expand(B, N_seg).reshape(B * N_seg)

        ctx = torch.no_grad() if self.freeze_lm else torch.enable_grad()
        with ctx:
            token_embeds = self.lm.token_embed(tokens_flat)
            pos_embeds = self.lm.pos_embed(positions_flat)
            species_embeds = self.lm.species_embed(species_flat).unsqueeze(1)

            x = token_embeds + pos_embeds + species_embeds

            for layer in self.lm.layers:
                x = layer(x, positions_flat, bucket_matrix=None)

            x = self.lm.final_norm(x)
            seg_repr = x.mean(dim=1)

        seg_repr = seg_repr.view(B, N_seg, -1)
        return seg_repr

    def forward(self, tokens, positions, species_ids):
        seg_repr = self.encode_segments(tokens, positions, species_ids)
        sample_repr = seg_repr.mean(dim=1)
        return self.classifier(sample_repr)


# ===========================================================
# 推理函数
# ===========================================================

def run_inference(model_path: Path, sequence: str, device: torch.device):
    print(f"[INFO] Loading model checkpoint: {model_path}")

    import torch.serialization
    torch.serialization.add_safe_globals([Config])
    torch.serialization.add_safe_globals([DNAModel])

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    config = ckpt["config"]
    species_list = ckpt["species_list"]
    label_map = ckpt["label_mapping"]
    freeze_lm = ckpt["freeze_lm"]

    lm = DNAModel(config, species_list=species_list)
    classifier = HierDNAClassifier(lm, num_classes=len(label_map), freeze_lm=freeze_lm)
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.to(device)
    classifier.eval()

    # 序列 → segments
    tokens = sequence_to_segments(sequence)     # [5000, 64]
    tokens = tokens.unsqueeze(0)                # [1, 5000, 64]

    L = tokens.size(-1)
    pos = torch.arange(L).view(1, 1, L).expand(1, 5000, L).clone()
    species_ids = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        logits = classifier(tokens, pos, species_ids)
        pred = logits.argmax(dim=-1).item()

    # id → label name
    inv_map = {v: k for k, v in label_map.items()}
    pred_label = inv_map[pred]

    return pred, pred_label


# ===========================================================
# main
# ===========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default='downstream/model/bailizhong_classifier.pth')
    p.add_argument("--sequence", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()

def select_device(x: str):
    if x == "cpu":
        return torch.device("cpu")
    if x == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if x == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}")

    pred_id, pred_label = run_inference(Path(args.model_path), args.sequence, device)
    print("\n==============================")
    print(f"Predicted class id   : {pred_id}")
    print(f"Predicted class name : {pred_label}")
    print("==============================\n")

if __name__ == "__main__":
    main()
