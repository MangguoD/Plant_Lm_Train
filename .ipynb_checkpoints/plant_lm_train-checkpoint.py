import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from Bio import SeqIO
import numpy as np
import math
import random
import os
from pathlib import Path
from einops import rearrange
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
import argparse
import psutil
import threading
from queue import Queue
import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # 让 cuDNN 自动选更快的算法


# ====================
# 配置参数
# ====================
class Config:
    # 模型容量
    d_model = 640
    n_buckets = 4096  # 6-mer 词表大小

    # 显存 & 吞吐主要看这几个
    batch_size = 160
    context_size = 128
    local_attn_window = 128

    learning_rate = 1e-4
    n_epochs = 10

    # 学习率 warmup（按 global_step）
    warmup_steps = 1000

    # 学习率衰减（按 epoch）
    lr_decay_start_epoch = 3
    lr_decay_factor = 0.5
    min_lr = 1e-6

    global_context = 200000  # 位置 embedding 上限
    n_heads = 10
    dropout = 0.1
    n_layers = 6
    temperature = 0.8

    chunk_size = 1200000     # 每次从基因组切出的碱基数

    # 桶相关配置
    bucket_norm_freq = 1000      # 每多少次 bucket 更新做一次归一化
    use_bucket_updater = True    # 开启桶机制

    save_dir = "dna_model"
    accumulation_steps = 1

    # DataLoader 设置
    num_workers = 8
    overlap_ratio = 0.5
    log_interval = 100       # 打印间隔（global_step）

    species_distribution = {
        "Glycine_max": 1.0,
        "Glycine_soja": 1.0
    }

    mixed_precision = True
    grad_clip = 1.0

    # 验证 / 测试最多跑多少个 batch（防止太慢）
    val_max_steps = 200

    # 本地 TensorBoard 日志目录
    tb_log_dir = "tb_logs"


# ====================
# 工具函数 & 数据集
# ====================
def six_mer_to_index(six_mer):
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    index = 0
    for i, base in enumerate(six_mer):
        index += base_map[base] * (4 ** (5 - i))
    return index


def index_to_six_mer(index):
    bases = ['A', 'C', 'G', 'T']
    six_mer = ""
    for _ in range(6):
        base_idx = index % 4
        six_mer = bases[base_idx] + six_mer
        index //= 4
    return six_mer


def clean_sequence(seq):
    return seq.upper().replace('N', '')


class GenomeIterableDataset(IterableDataset):
    """
    按物种 + 文件顺序扫描生成样本。
    通过 split 标记在样本级别做 8:1:1 的 train/val/test 切分。
    """
    def __init__(self, species_files, config: Config, split: str = "train"):
        assert split in ("train", "val", "test")
        self.config = config
        self.species_files = species_files
        self.species_list = list(species_files.keys())
        self.split = split

        # 固定 8:1:1 切分比例
        self.split_bucket = {
            "train": (0, 8),
            "val": (8, 9),
            "test": (9, 10)
        }

        # 粗略估计数据量（__len__ 用，近似即可）
        base_total_chunks = 0
        for _, files in species_files.items():
            for f in files:
                file_size = os.path.getsize(f)
                base_total_chunks += file_size // (config.chunk_size * 2)

        ratio_map = {"train": 0.8, "val": 0.1, "test": 0.1}
        self.total_chunks = int(base_total_chunks * ratio_map[self.split])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_species = self.species_list[worker_id::num_workers]
        if len(worker_species) == 0:
            return

        rng = np.random.default_rng()
        species_order = list(worker_species)
        rng.shuffle(species_order)

        for species in species_order:
            files = list(self.species_files[species])
            rng.shuffle(files)

            for file_path in files:
                with open(file_path, "r") as f:
                    seq_accumulator = ""
                    for record in SeqIO.parse(f, "fasta"):
                        seq = clean_sequence(str(record.seq))
                        seq_accumulator += seq

                        # 按 chunk_size 切块
                        while len(seq_accumulator) >= self.config.chunk_size:
                            chunk = seq_accumulator[:self.config.chunk_size]
                            seq_accumulator = seq_accumulator[self.config.chunk_size:]

                            # 从这个 chunk 生成多个重叠样本
                            yield from self.generate_samples(chunk, species)

    def generate_samples(self, chunk, species):
        tokens = []
        for i in range(0, len(chunk) - 5, 6):
            token = chunk[i:i + 6]
            if len(token) == 6 and all(b in 'ACGT' for b in token):
                tokens.append(six_mer_to_index(token))

        step = int(self.config.context_size * (1 - self.config.overlap_ratio))
        step = max(step, 1)

        bucket_start, bucket_end = self.split_bucket[self.split]
        sample_idx = 0

        for start in range(0, len(tokens) - self.config.context_size, step):
            bucket = sample_idx % 10
            sample_idx += 1

            if not (bucket_start <= bucket < bucket_end):
                continue

            end = start + self.config.context_size
            yield {
                "tokens": tokens[start:end],
                "positions": list(range(start, end)),
                "targets": tokens[start + 1:end + 1],
                "species": species
            }

    def __len__(self):
        return self.total_chunks * (self.config.chunk_size // (self.config.context_size * 3))


def genome_collate_fn(batch):
    tokens_list = [torch.tensor(sample["tokens"], dtype=torch.long) for sample in batch]
    positions_list = [torch.tensor(sample["positions"], dtype=torch.long) for sample in batch]
    targets_list = [torch.tensor(sample["targets"], dtype=torch.long) for sample in batch]

    tokens = torch.stack(tokens_list, dim=0)
    positions = torch.stack(positions_list, dim=0)
    targets = torch.stack(targets_list, dim=0)
    species = [sample["species"] for sample in batch]

    return {
        "tokens": tokens,
        "positions": positions,
        "targets": targets,
        "species": species
    }


# ====================
# 异步桶更新（目前未在训练中使用线程）
# ====================
class BucketUpdater:
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.queue = Queue(maxsize=10000)
        self.running = True
        self.update_count = 0
        self.thread = threading.Thread(target=self.update_loop)
        self.thread.daemon = True
        self.thread.start()

    def update_loop(self):
        print("Bucket updater started", flush=True)
        while self.running:
            batch_data = self.queue.get()
            if batch_data is None:
                break

            tokens, positions, species = batch_data
            with torch.no_grad():
                self.model.update_bucket(tokens, positions)
                self.update_count += 1

                if self.update_count % self.config.bucket_norm_freq == 0:
                    self.model.normalize_buckets()
                    if dist.is_initialized() and dist.get_rank() == 0:
                        print(f"Normalized buckets at update {self.update_count}", flush=True)

    def add_batch(self, tokens, positions, species):
        if self.queue.full():
            _ = self.queue.get()
        self.queue.put((tokens, positions, species))

    def shutdown(self):
        self.running = False
        self.queue.put(None)
        self.thread.join()
        print("Bucket updater stopped", flush=True)


# ====================
# 注意力模块
# ====================
class CausalLocalAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", self.create_causal_mask(window_size))

    def create_causal_mask(self, size):
        mask = torch.tril(torch.ones(size, size))
        return mask.bool()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = rearrange(self.q_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)

        num_blocks = (seq_len + self.window_size - 1) // self.window_size
        padding = num_blocks * self.window_size - seq_len

        q = F.pad(q, (0, 0, 0, padding))
        k = F.pad(k, (0, 0, 0, padding))
        v = F.pad(v, (0, 0, 0, padding))

        q_blocks = rearrange(q, 'b h (n w) d -> b h n w d', w=self.window_size)
        k_blocks = rearrange(k, 'b h (n w) d -> b h n w d', w=self.window_size)
        v_blocks = rearrange(v, 'b h (n w) d -> b h n w d', w=self.window_size)

        attn_scores = torch.einsum(
            'bhnqd,bhnkd->bhnqk', q_blocks, k_blocks
        ) / math.sqrt(self.d_head)

        mask = self.mask[:self.window_size, :self.window_size]
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(attn_scores.device)
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = torch.where(mask, attn_scores, mask_value)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum('bhnqk,bhnkd->bhnqd', attn_weights, v_blocks)
        attn_output = rearrange(attn_output, 'b h n w d -> b h (n w) d')
        attn_output = attn_output[:, :, :seq_len, :]

        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)


class GlobalAttention(nn.Module):
    """
    全局桶注意力：使用共享 bucket_matrix: [n_buckets, d_model]。
    """
    def __init__(self, d_model, n_heads, n_buckets, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, bucket_matrix):
        """
        x: [B, S, D]
        bucket_matrix: [N_buckets, D]
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)

        kv = self.kv_proj(bucket_matrix)       # [N_buckets, 2D]
        k, v = torch.chunk(kv, 2, dim=-1)      # [N_buckets, D] x2

        k = rearrange(k, 'n (h d) -> 1 h n d', h=self.n_heads).expand(batch_size, -1, -1, -1)
        v = rearrange(v, 'n (h d) -> 1 h n d', h=self.n_heads).expand(batch_size, -1, -1, -1)

        attn_scores = torch.einsum('bhsd,bhnd->bhsn', q, k) / math.sqrt(self.d_head)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum('bhsn,bhnd->bhsd', attn_weights, v)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)


class DNALayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.local_attn = CausalLocalAttention(
            config.d_model,
            config.n_heads,
            config.local_attn_window,
            dropout=config.dropout
        )

        self.global_attn = GlobalAttention(
            config.d_model,
            config.n_heads,
            config.n_buckets,
            dropout=config.dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout)
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

    def forward(self, x, positions, bucket_matrix=None):
        # 局部注意力
        local_out = self.local_attn(x)
        x = self.norm1(x + local_out)

        # 全局桶注意力
        if bucket_matrix is not None:
            global_out = self.global_attn(x, bucket_matrix)
            x = self.norm2(x + global_out)
        else:
            x = self.norm2(x)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class DNAModel(nn.Module):
    def __init__(self, config: Config, species_list):
        super().__init__()
        self.config = config
        self.species_list = species_list

        self.token_embed = nn.Embedding(config.n_buckets, config.d_model)
        self.pos_embed = nn.Embedding(config.global_context, config.d_model)
        self.species_embed = nn.Embedding(len(species_list), config.d_model)

        # ---- 关键改动：bucket_matrix / bucket_count 作为 buffer，不参与 AdamW ----
        self.register_buffer(
            "bucket_matrix",
            torch.zeros(config.n_buckets, config.d_model)
        )
        self.register_buffer(
            "bucket_count",
            torch.zeros(config.n_buckets, dtype=torch.float32)
        )

        self.layers = nn.ModuleList([
            DNALayer(config) for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.n_buckets)

        self.apply(self._init_weights)

        # 权重共享：输出层权重 = token embedding
        self.classifier.weight = self.token_embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def update_bucket(self, token_indices, token_positions, momentum: float = 0.001):
        """
        用当前位置 embedding 对 bucket 做滑动平均更新：
            m_new = (1 - momentum) * m_old + momentum * pos_vec
        token_indices: [B, L]
        token_positions: [B, L]
        """
        device = self.pos_embed.weight.device
        token_indices = token_indices.to(device=device, dtype=torch.long)
        token_positions = token_positions.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # 位置向量 [B, L, D]
            pos_vecs = self.pos_embed(token_positions)
            # 每个样本平均一下 [B, D]
            pos_vecs = pos_vecs.mean(dim=1)

            B, L = token_indices.shape
            flat_index = token_indices.view(-1)               # [B*L]
            flat_pos = pos_vecs.repeat_interleave(L, dim=0)   # [B*L, D]

            # unique 聚合到每个 bucket
            uniq_index, inverse = torch.unique(flat_index, return_inverse=True)  # [K], [B*L]

            bucket_updates = torch.zeros(
                uniq_index.size(0),
                self.config.d_model,
                device=device
            )
            bucket_updates.index_add_(0, inverse, flat_pos)

            counts = torch.bincount(inverse, minlength=uniq_index.size(0)).float().unsqueeze(1)
            bucket_updates = bucket_updates / counts  # [K, D]

            old_vecs = self.bucket_matrix[uniq_index]  # [K, D]
            new_vecs = (1.0 - momentum) * old_vecs + momentum * bucket_updates
            self.bucket_matrix[uniq_index] = new_vecs

            # 如果你以后要按频次做别的事情，可以用 bucket_count
            self.bucket_count[uniq_index] += counts.squeeze(1)

    def normalize_buckets(self):
        with torch.no_grad():
            norms = torch.norm(self.bucket_matrix, p=2, dim=1, keepdim=True)
            norms = torch.where(norms > 0, norms, torch.ones_like(norms))
            self.bucket_matrix.div_(norms)

    def forward(self, input_tokens, token_positions, species_ids):
        token_embeds = self.token_embed(input_tokens)
        pos_embeds = self.pos_embed(token_positions)
        species_embeds = self.species_embed(species_ids).unsqueeze(1)

        x = token_embeds + pos_embeds + species_embeds

        bucket_matrix = self.bucket_matrix

        for layer in self.layers:
            x = layer(x, token_positions, bucket_matrix=bucket_matrix)

        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits


# ====================
# 分布式辅助
# ====================
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ====================
# 验证 / 测试逻辑
# ====================
def evaluate(rank,
             model,
             dataloader,
             config: Config,
             species_list,
             global_step,
             writer=None,
             split_name: str = "val"):
    if dataloader is None:
        return float("nan")

    model.eval()
    device = torch.device(f"cuda:{rank}")

    total_loss = 0.0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= config.val_max_steps:
                break

            tokens = batch["tokens"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            species_ids = torch.tensor(
                [species_list.index(s) for s in batch["species"]],
                dtype=torch.long,
                device=device
            )

            logits = model(tokens, positions, species_ids)
            loss = criterion(
                logits.view(-1, config.n_buckets),
                targets.view(-1)
            )

            total_loss += loss.item()
            total_tokens += targets.numel()

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
    else:
        avg_loss = float("nan")
        ppl = float("nan")

    if rank == 0:
        print(
            f"[Eval-{split_name} | S{global_step:06d}] "
            f"{split_name}_token_loss={avg_loss:.4f} | {split_name}_ppl={ppl:.2f}",
            flush=True
        )
        if writer is not None:
            writer.add_scalar(f"{split_name}/token_loss", avg_loss, global_step)
            writer.add_scalar(f"{split_name}/ppl", ppl, global_step)

    model.train()
    torch.cuda.empty_cache()
    return avg_loss


# ====================
# 训练 Worker
# ====================
def train_worker(rank, world_size, config: Config, species_files):
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    writer = SummaryWriter(log_dir=config.tb_log_dir) if rank == 0 else None

    # ========= 构建 train / val / test 数据集 =========
    train_dataset = GenomeIterableDataset(species_files, config, split="train")

    max_workers = config.num_workers
    num_species = len(train_dataset.species_list)
    effective_workers = min(max_workers, num_species)
    if effective_workers < 1:
        effective_workers = 1

    print(
        f"[rank {rank}] Using {effective_workers} dataloader workers for {num_species} train species",
        flush=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=genome_collate_fn,
        prefetch_factor=2
    )

    # 验证集 / 测试集仅在 rank0 构建和使用
    if rank == 0:
        val_dataset = GenomeIterableDataset(species_files, config, split="val")
        test_dataset = GenomeIterableDataset(species_files, config, split="test")

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=min(config.num_workers, len(val_dataset.species_list)),
            pin_memory=True,
            persistent_workers=False,
            collate_fn=genome_collate_fn,
            prefetch_factor=2
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=min(config.num_workers, len(test_dataset.species_list)),
            pin_memory=True,
            persistent_workers=False,
            collate_fn=genome_collate_fn,
            prefetch_factor=2
        )

        print("[rank 0] Val/Test split enabled with 8:1:1 ratio at sample level.", flush=True)
    else:
        val_dataloader = None
        test_dataloader = None

    species_list = list(species_files.keys())
    model = DNAModel(config, species_list).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )

    scaler = GradScaler(enabled=config.mixed_precision)

    global_step = 0
    best_val_loss = float("inf") if rank == 0 else None

    # 统计 bucket 更新次数（用来触发 normalize）
    bucket_update_count = 0

    for epoch in range(config.n_epochs):
        model.train()
        if rank == 0:
            print(f"\n===== Epoch {epoch} started =====", flush=True)

        # epoch 级 LR 基准
        if epoch < config.lr_decay_start_epoch:
            epoch_base_lr = config.learning_rate
        else:
            decay_steps = epoch - config.lr_decay_start_epoch + 1
            epoch_base_lr = config.learning_rate * (config.lr_decay_factor ** decay_steps)
            epoch_base_lr = max(epoch_base_lr, config.min_lr)

        if rank == 0:
            print(f"[Epoch {epoch}] base_lr={epoch_base_lr:.6e}", flush=True)

        epoch_loss = 0.0
        epoch_samples = 0
        start_time = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader):
            tokens = batch["tokens"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            species_ids = torch.tensor(
                [species_list.index(s) for s in batch["species"]],
                dtype=torch.long,
                device=device
            )

            with autocast(device_type="cuda", enabled=config.mixed_precision):
                logits = model(tokens, positions, species_ids)
                loss = criterion(
                    logits.view(-1, config.n_buckets),
                    targets.view(-1)
                )
                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1

                # global_step 级 warmup
                if global_step <= config.warmup_steps:
                    lr_scale = float(global_step) / float(max(1, config.warmup_steps))
                else:
                    lr_scale = 1.0

                current_lr = epoch_base_lr * lr_scale

                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                if rank == 0 and writer is not None:
                    writer.add_scalar("train/lr", current_lr, global_step)

                # ======= 桶机制：同步更新 bucket_matrix =======
                if config.use_bucket_updater:
                    with torch.no_grad():
                        model.module.update_bucket(
                            tokens.detach(),
                            positions.detach(),
                            momentum=0.001
                        )
                        bucket_update_count += 1

                        if bucket_update_count % config.bucket_norm_freq == 0:
                            model.module.normalize_buckets()
                            if rank == 0:
                                print(
                                    f"[Bucket] normalize at global_step={global_step}, "
                                    f"updates={bucket_update_count}",
                                    flush=True
                                )

            epoch_loss += loss.item() * config.accumulation_steps
            epoch_samples += tokens.size(0)

            # 日志
            if rank == 0 and (global_step > 0 and global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                samples_per_sec = epoch_samples / elapsed if elapsed > 0 else 0
                mem_usage = psutil.virtual_memory().percent
                avg_loss = epoch_loss / (batch_idx + 1)

                print(
                    f"[E{epoch:02d} | S{global_step:06d}] "
                    f"loss={avg_loss:.4f} | "
                    f"speed={samples_per_sec:.1f} samples/s | "
                    f"mem={mem_usage:.1f}%",
                    flush=True
                )

                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/throughput", samples_per_sec, global_step)
                    writer.add_scalar("system/memory_percent", mem_usage, global_step)

        # ====== 每个 epoch 结束后：rank0 做一次验证 + 保存 ckpt / best_model ======
        if rank == 0:
            val_loss = evaluate(
                rank=rank,
                model=model,
                dataloader=val_dataloader,
                config=config,
                species_list=species_list,
                global_step=global_step,
                writer=writer,
                split_name="val"
            )

            save_path = Path(config.save_dir) / f"model_epoch_{epoch}.pt"
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'config': config
            }, save_path)
            print(f"[rank 0] Saved checkpoint to {save_path}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = Path(config.save_dir) / "model_best.pt"
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': config,
                    'best_val_loss': best_val_loss
                }, best_path)
                print(f"[rank 0] New best model saved to {best_path}", flush=True)

    # 全部训练结束后，rank0 额外做一次 test 评估
    if rank == 0 and test_dataloader is not None:
        _ = evaluate(
            rank=rank,
            model=model,
            dataloader=test_dataloader,
            config=config,
            species_list=species_list,
            global_step=global_step,
            writer=writer,
            split_name="test"
        )

    if writer is not None:
        writer.close()

    cleanup_distributed()


# ====================
# main
# ====================
def main():
    parser = argparse.ArgumentParser(description='Multi-species Genome Training')
    parser.add_argument('--world_size', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing species data')
    args = parser.parse_args()

    config = Config()

    data_dir = Path(args.data_dir)

    # ========= 收集所有物种的 fasta 文件 =========
    species_files = {}

    for species_dir in data_dir.iterdir():
        if species_dir.is_dir():
            species_name = species_dir.name
            fasta_files = list(species_dir.glob("*.fna")) + list(species_dir.glob("*.fa"))
            fasta_files = [str(f) for f in fasta_files]
            if not fasta_files:
                continue

            species_files[species_name] = fasta_files

            print(
                f"Species {species_name}: total_files={len(fasta_files)}",
                flush=True
            )

    if not species_files:
        raise ValueError("No species data found in the specified directory")

    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.tb_log_dir).mkdir(parents=True, exist_ok=True)

    torch.multiprocessing.spawn(
        train_worker,
        args=(args.world_size, config, species_files),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    main()