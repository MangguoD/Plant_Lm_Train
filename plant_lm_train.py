import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import math
import random
import os
import glob
from pathlib import Path
from einops import rearrange
import gc
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import argparse
import psutil
import threading
from queue import Queue
import wandb


# 配置参数
class Config:
    d_model = 512
    n_buckets = 4096  # 桶数量
    batch_size = 16  # 批大小
    learning_rate = 1e-4
    n_epochs = 10
    context_size = 64  # 训练上下文长度(tokens)
    global_context = 200000  # 全局上下文长度(20万tokens)
    n_heads = 8  # 注意力头数
    local_attn_window = 32  # 局部注意力窗口大小
    dropout = 0.1  # 正则化
    n_layers = 6  # 层数
    temperature = 0.8  # 生成温度
    chunk_size = 1200000  # 序列处理块大小(碱基数)
    bucket_norm_freq = 1000  # 桶矩阵归一化频率
    save_dir = "dna_model"  # 模型保存目录
    accumulation_steps = 4  # 梯度累积步数
    num_workers = 4  # 每GPU数据工作进程数
    overlap_ratio = 0.5  # 重叠采样比例
    log_interval = 100  # 日志记录间隔
    species_distribution = {  # 物种采样权重
        "glycine_max": 1.0,
        "arabidopsis": 0.8,
        "oryza_sativa": 0.7,
        "zea_mays": 0.6,
        "vitis_vinifera": 0.5
    }
    mixed_precision = True  # 启用混合精度训练


# 碱基映射
base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
int_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


# 6-mer到索引的映射
def six_mer_to_index(six_mer):
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    index = 0
    for i, base in enumerate(six_mer):
        index += base_map[base] * (4 ** (5 - i))
    return index


# 索引到6-mer的映射
def index_to_six_mer(index):
    bases = ['A', 'C', 'G', 'T']
    six_mer = ""
    for i in range(6):
        base_idx = index % 4
        six_mer = bases[base_idx] + six_mer
        index = index // 4
    return six_mer


def clean_sequence(seq):
    return seq.upper().replace('N', '')


class GenomeIterableDataset(IterableDataset):

    def __init__(self, species_files, config):
        self.config = config
        self.species_files = species_files
        self.species_list = list(species_files.keys())
        self.species_weights = [
            config.species_distribution[s] for s in self.species_list
        ]

        # 计算总chunk数（近似值）
        self.total_chunks = 0
        for species, files in species_files.items():
            for f in files:
                file_size = os.path.getsize(f)
                self.total_chunks += file_size // (config.chunk_size * 2)

        self.current_weights = self.species_weights.copy()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        worker_species = self.species_list[worker_id::num_workers]
        worker_weights = [self.species_distribution[s] for s in worker_species]

        # 重置当前权重
        self.current_weights = worker_weights.copy()

        while True:
            # 根据权重选择物种
            total_weight = sum(self.current_weights)
            probs = [w / total_weight for w in self.current_weights]
            species_idx = np.random.choice(len(worker_species), p=probs)
            species = worker_species[species_idx]

            # 降低当前权重（确保平衡）
            self.current_weights[species_idx] *= 0.8
            if min(self.current_weights) < 0.1:
                self.current_weights = worker_weights.copy()

            file_path = random.choice(self.species_files[species])

            # 流式读取FASTA
            with open(file_path, "r") as f:
                seq_accumulator = ""
                for record in SeqIO.parse(f, "fasta"):
                    # 过滤并处理序列
                    seq = clean_sequence(str(record.seq))
                    seq_accumulator += seq

                    # 分割为chunks
                    while len(seq_accumulator) >= self.config.chunk_size:
                        chunk = seq_accumulator[:self.config.chunk_size]
                        seq_accumulator = seq_accumulator[self.config.chunk_size:]

                        # 生成重叠样本
                        yield from self.generate_samples(chunk, species)

    def generate_samples(self, chunk, species):
        """从单个chunk生成重叠样本"""
        tokens = []
        # 流式tokenization
        for i in range(0, len(chunk) - 5, 6):
            token = chunk[i:i + 6]
            if len(token) == 6 and all(b in 'ACGT' for b in token):
                tokens.append(six_mer_to_index(token))

        # 重叠采样
        step = int(self.config.context_size * (1 - self.config.overlap_ratio))
        step = max(step, 1)  # 确保至少为1

        # 生成样本窗口
        for start in range(0, len(tokens) - self.config.context_size, step):
            end = start + self.config.context_size
            yield {
                "tokens": tokens[start:end],
                "positions": list(range(start, end)),
                "targets": tokens[start + 1:end + 1],
                "species": species
            }

    def __len__(self):
        """近似长度，用于进度条等"""
        return self.total_chunks * (self.config.chunk_size // (self.config.context_size * 3))


class BucketUpdater:
    """桶矩阵异步更新器"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.queue = Queue(maxsize=10000)
        self.running = True
        self.update_count = 0
        self.thread = threading.Thread(target=self.update_loop)
        self.thread.daemon = True
        self.thread.start()

    def update_loop(self):
        """后台更新线程"""
        print("Bucket updater started")
        while self.running:
            batch_data = self.queue.get()
            if batch_data is None:  # 终止信号
                break

            tokens, positions, species = batch_data
            with torch.no_grad():
                # 更新桶矩阵
                self.model.update_bucket(tokens, positions, species)
                self.update_count += 1

                # 定期归一化
                if self.update_count % self.config.bucket_norm_freq == 0:
                    self.model.normalize_buckets(species)
                    if dist.is_initialized() and dist.get_rank() == 0:
                        print(f"Normalized buckets for {species} at update {self.update_count}")

    def add_batch(self, tokens, positions, species):
        """添加更新批次"""
        if self.queue.full():
            # 队列满时丢弃最旧的数据
            _ = self.queue.get()
        self.queue.put((tokens, positions, species))

    def shutdown(self):
        """关闭更新器"""
        self.running = False
        self.queue.put(None)
        self.thread.join()
        print("Bucket updater stopped")


class CausalLocalAttention(nn.Module):
    """因果局部注意力（只能看到前面的token）"""

    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.window_size = window_size

        # 线性变换
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 因果掩码
        self.register_buffer("mask", self.create_causal_mask(window_size))

    def create_causal_mask(self, size):
        """创建因果掩码（只能看到前面的token）"""
        mask = torch.tril(torch.ones(size, size))
        return mask.bool()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = rearrange(self.q_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)

        # 分块处理 - 使用滑动窗口
        num_blocks = (seq_len + self.window_size - 1) // self.window_size
        padding = num_blocks * self.window_size - seq_len

        # 填充序列
        q = F.pad(q, (0, 0, 0, padding))
        k = F.pad(k, (0, 0, 0, padding))
        v = F.pad(v, (0, 0, 0, padding))

        # 分块
        q_blocks = rearrange(q, 'b h (n w) d -> b h n w d', w=self.window_size)
        k_blocks = rearrange(k, 'b h (n w) d -> b h n w d', w=self.window_size)
        v_blocks = rearrange(v, 'b h (n w) d -> b h n w d', w=self.window_size)

        # 计算注意力分数
        attn_scores = torch.einsum('bhnqd,bhnkd->bhnqk', q_blocks, k_blocks) / math.sqrt(self.d_head)

        mask = self.mask[:self.window_size, :self.window_size]
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,window,window]
        mask = mask.to(attn_scores.device)
        mask_value = torch.finfo(attn_scores.dtype).min

        attn_scores = torch.where(
            mask,
            attn_scores,
            mask_value
        )

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.einsum('bhnqk,bhnkd->bhnqd', attn_weights, v_blocks)
        attn_output = rearrange(attn_output, 'b h n w d -> b h (n w) d')

        attn_output = attn_output[:, :, :seq_len, :]

        # 合并多头输出
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)


class GlobalAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_buckets, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)

        self.kv_proj = nn.Linear(d_model, 2 * d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, bucket_matrix):
        """x: [batch, seq_len, d_model]"""
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)

        # 投影桶矩阵
        kv = self.kv_proj(bucket_matrix)  # [n_buckets, 2*d_model]
        k, v = torch.chunk(kv, 2, dim=-1)  # [n_buckets, d_model]

        # 扩展维度用于批处理
        k = rearrange(k, 'n (h d) -> 1 h n d', h=self.n_heads)
        v = rearrange(v, 'n (h d) -> 1 h n d', h=self.n_heads)
        k = k.expand(batch_size, -1, -1, -1)
        v = v.expand(batch_size, -1, -1, -1)

        # 计算注意力分数
        attn_scores = torch.einsum('bhsd,bhnd->bhsn', q, k) / math.sqrt(self.d_head)

        # 应用注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.einsum('bhsn,bhnd->bhsd', attn_weights, v)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)


class DNALayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 局部注意力
        self.local_attn = CausalLocalAttention(
            config.d_model,
            config.n_heads,
            config.local_attn_window,
            dropout=config.dropout
        )

        # 全局桶注意力
        self.global_attn = GlobalAttention(
            config.d_model,
            config.n_heads,
            config.n_buckets,
            dropout=config.dropout
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

    def forward(self, x, positions, bucket_matrix):
        """
        x: [batch, seq_len, d_model]
        positions: [batch, seq_len]
        bucket_matrix: [n_buckets, d_model] 桶矩阵
        """
        attn_out = self.local_attn(x)
        x = self.norm1(x + attn_out)

        global_out = self.global_attn(x, bucket_matrix)
        x = self.norm2(x + global_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class DNAModel(nn.Module):

    def __init__(self, config, species_list):
        super().__init__()
        self.config = config
        self.species_list = species_list

        self.token_embed = nn.Embedding(config.n_buckets, config.d_model)

        self.pos_embed = nn.Embedding(config.global_context, config.d_model)

        self.species_embed = nn.Embedding(len(species_list), config.d_model)

        self.bucket_matrices = nn.ParameterDict({
            species: nn.Parameter(torch.zeros(config.n_buckets, config.d_model))
            for species in species_list
        })

        # 模型层
        self.layers = nn.ModuleList([
            DNALayer(config) for _ in range(config.n_layers)
        ])

        # 最终归一化
        self.final_norm = nn.LayerNorm(config.d_model)

        self.classifier = nn.Linear(config.d_model, config.n_buckets)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_bucket_matrix(self, species):
        return self.bucket_matrices[species]

    def update_bucket(self, token_indices, token_positions, species):
        """更新桶矩阵"""
        # 获取位置向量
        pos_vecs = self.pos_embed(token_positions)

        bucket_matrix = self.bucket_matrices[species]
        bucket_matrix.index_add_(
            dim=0,
            index=token_indices,
            source=pos_vecs
        )

    def normalize_buckets(self, species):
        """归一化指定物种的桶矩阵"""
        bucket_matrix = self.bucket_matrices[species]
        with torch.no_grad():
            norms = torch.norm(bucket_matrix, p=2, dim=1, keepdim=True)
            norms = torch.where(norms > 0, norms, torch.ones_like(norms))
            bucket_matrix.div_(norms)

    def forward(self, input_tokens, token_positions, species_ids):
        """
        input_tokens: [batch, seq_len] 输入token序列
        token_positions: [batch, seq_len] 对应位置索引
        species_ids: [batch] 物种ID
        """
        # 1. 嵌入token和位置
        token_embeds = self.token_embed(input_tokens)
        pos_embeds = self.pos_embed(token_positions)

        # 2. 嵌入物种信息
        species_embeds = self.species_embed(species_ids).unsqueeze(1)  # [batch, 1, d_model]

        # 3. 组合嵌入
        x = token_embeds + pos_embeds + species_embeds

        species_name = self.species_list[species_ids[0].item()]
        bucket_matrix = self.get_bucket_matrix(species_name)

        for layer in self.layers:
            x = layer(x, token_positions, bucket_matrix)

        x = self.final_norm(x)

        logits = self.classifier(x)  # [batch, seq_len, n_buckets]

        return logits


def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def train_worker(rank, world_size, config, species_files):

    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        wandb.init(project="genome-transformer", config=vars(config))

    # 创建数据集
    dataset = GenomeIterableDataset(species_files, config)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # 数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # 初始化模型
    species_list = list(species_files.keys())
    model = DNAModel(config, species_list).to(device)
    model = DDP(model, device_ids=[rank])

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    # 混合精度训练
    scaler = GradScaler(enabled=config.mixed_precision)

    # 桶矩阵更新器
    if rank == 0:
        bucket_updater = BucketUpdater(model.module, config)
    else:
        bucket_updater = None

    global_step = 0
    for epoch in range(config.n_epochs):
        model.train()
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_samples = 0
        start_time = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):

            tokens = batch["tokens"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            species_ids = torch.tensor([
                species_list.index(s) for s in batch["species"]
            ], dtype=torch.long)


            with autocast(enabled=config.mixed_precision):

                logits = model(tokens, positions, species_ids)

                loss = criterion(
                    logits.view(-1, config.n_buckets),
                    targets.view(-1)
                )

                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            # 桶矩阵更新（异步）
            if rank == 0 and batch_idx % config.accumulation_steps == 0:
                bucket_updater.add_batch(
                    tokens.cpu().view(-1),
                    positions.cpu().view(-1),
                    batch["species"][0]
                )

            if (batch_idx + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1

            epoch_loss += loss.item() * config.accumulation_steps
            epoch_samples += tokens.size(0)

            if global_step % config.log_interval == 0 and rank == 0:

                elapsed = time.time() - start_time
                samples_per_sec = epoch_samples / elapsed if elapsed > 0 else 0
                mem_usage = psutil.virtual_memory().percent

                wandb.log({
                    "loss": epoch_loss / (batch_idx + 1),
                    "throughput": samples_per_sec,
                    "memory": mem_usage,
                    "epoch": epoch,
                    "step": global_step
                })

                print(f"Epoch {epoch}, Step {global_step}: Loss={epoch_loss / (batch_idx + 1):.4f}, "
                      f"Throughput={samples_per_sec:.1f} samples/sec, Mem={mem_usage}%")

        if rank == 0:
            save_path = Path(config.save_dir) / f"model_epoch_{epoch}.pt"
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'config': config
            }, save_path)

    if rank == 0:
        bucket_updater.shutdown()


    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Multi-species Genome Training')
    parser.add_argument('--world_size', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing species data')
    args = parser.parse_args()

    config = Config()

    species_files = {}
    data_dir = Path(args.data_dir)

    for species_dir in data_dir.iterdir():
        if species_dir.is_dir():
            species_name = species_dir.name
            fasta_files = list(species_dir.glob("*.fna")) + list(species_dir.glob("*.fa"))
            if fasta_files:
                species_files[species_name] = [str(f) for f in fasta_files]
                print(f"Found {len(fasta_files)} files for {species_name}")

    if not species_files:
        raise ValueError("No species data found in the specified directory")

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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