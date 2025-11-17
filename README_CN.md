# Plant Language Model 训练系统

## 项目简介

本项目实现了一个基于Transformer的多物种植物基因组语言模型训练系统。模型使用6-mer tokenization策略，结合局部-全局注意力机制，支持超长序列的基因组数据处理。

## 核心特性

### 模型架构

- **6-mer Tokenization**: 将DNA序列按6个碱基为单位编码（词汇表大小: 4^6 = 4096）
- **双层注意力机制**:
  - 局部注意力: 因果滑动窗口（32 tokens），捕获短程依赖
  - 全局注意力: 基于桶矩阵的压缩记忆（4096 buckets），处理长程依赖
- **多物种支持**: 通过物种嵌入层区分不同物种的基因组特征
- **长序列处理**: 支持20万token的全局上下文（相当于120万碱基对）

### 训练优化

- **分布式训练**: 支持多GPU并行（PyTorch DDP）
- **混合精度**: 使用Automatic Mixed Precision降低显存占用
- **梯度累积**: 减少batch size对显存的压力
- **流式数据加载**: 无需一次性加载全基因组到内存
- **动态桶更新**: 异步更新全局记忆矩阵

### 数据处理

- **重叠采样**: 生成训练样本时使用50%重叠，增加数据利用率
- **物种平衡采样**: 根据配置权重动态平衡不同物种的采样比例
- **序列过滤**: 自动过滤N字符和非标准碱基
- **块化处理**: 将长染色体分割为120万bp的chunk进行处理

## 技术实现细节

### Token表示

每个6-mer序列映射到一个0-4095的整数:
```
index = base[0] * 4^5 + base[1] * 4^4 + ... + base[5] * 4^0
其中 A=0, C=1, G=2, T=3
```

### 桶矩阵机制

全局桶矩阵(bucket matrix)为每个token维护位置信息的聚合:
- 维度: [4096 buckets × 512 d_model]
- 更新策略: 每1000步归一化一次
- 作用: 压缩长序列历史信息，提供全局上下文

### 注意力计算

**局部注意力**:
- 窗口大小: 32 tokens
- 因果掩码: 只能看到当前及之前的token
- 分块计算: 减少显存占用

**全局注意力**:
- Query来自当前序列
- Key/Value来自桶矩阵
- 每个位置都能访问全部4096个桶

### 训练流程

1. 数据加载器流式读取FASTA文件
2. 序列清洗（移除N字符）
3. 6-mer tokenization
4. 生成重叠训练样本
5. 前向传播计算损失
6. 反向传播更新参数
7. 异步更新桶矩阵

## 系统要求

### 硬件配置

**最低配置**:
- CPU: 4核
- 内存: 16GB
- 存储: 50GB SSD

**推荐配置**:
- GPU: NVIDIA V100/A100/RTX 3090 (16GB+ 显存)
- CPU: 16核+
- 内存: 32GB+
- 存储: 100GB+ SSD

### 软件依赖

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (使用GPU时)
- BioPython 1.81+
- 其他依赖见 requirements.txt

## 快速开始

### 1. 环境配置

```bash
bash setup_env.sh
```

自动完成虚拟环境创建和依赖安装。

### 2. 数据准备

组织数据目录结构:
```
data/
├── species_1/
│   ├── chr1.fna
│   └── chr2.fna
└── species_2/
    └── genome.fa
```

支持的格式: .fna, .fa, .fasta

### 3. 启动训练

```bash
bash run.sh
```

## 配置参数

### 模型超参数 (plant_lm_train.py - Config类)

```python
d_model = 512              # 模型维度
n_buckets = 4096          # 桶数量(等于词汇表大小)
batch_size = 16           # 批大小
learning_rate = 1e-4      # 学习率
n_epochs = 10             # 训练轮数
context_size = 64         # 训练序列长度(tokens)
global_context = 200000   # 全局上下文容量
n_heads = 8               # 注意力头数
local_attn_window = 32    # 局部窗口大小
n_layers = 6              # Transformer层数
dropout = 0.1             # Dropout率
chunk_size = 1200000      # 序列块大小(bp)
overlap_ratio = 0.5       # 重叠采样比例
accumulation_steps = 4    # 梯度累积步数
```

### 物种采样权重

```python
species_distribution = {
    "glycine_max": 1.0,     # 大豆
    "arabidopsis": 0.8,     # 拟南芥
    "oryza_sativa": 0.7,    # 水稻
    "zea_mays": 0.6,        # 玉米
    "vitis_vinifera": 0.5   # 葡萄
}
```

### 运行时参数 (run.sh)

```bash
DATA_DIR="./data"          # 数据目录
WORLD_SIZE=2               # GPU数量
WANDB_MODE="online"        # 实验跟踪模式
```

## 输出说明

### 模型检查点

位置: `dna_model/model_epoch_N.pt`

包含内容:
- model_state_dict: 模型权重
- optimizer_state_dict: 优化器状态
- epoch: 当前轮次
- config: 训练配置

### 训练日志

位置: `logs/training_TIMESTAMP.log`

记录内容:
- 每步的损失值
- 吞吐量(samples/sec)
- 内存使用
- GPU状态

## 性能优化建议

### 显存优化

- 减小 `batch_size`
- 减小 `context_size`
- 减少 `num_workers`
- 启用梯度累积

### 速度优化

- 增加 `batch_size`（在显存允许范围内）
- 使用更多GPU（增加 `WORLD_SIZE`）
- 使用SSD存储数据
- 增加 `num_workers`

### 训练稳定性

- 调整 `learning_rate`
- 使用学习率调度器
- 增加梯度裁剪阈值
- 监控损失曲线

## 常见问题

### CUDA不可用

检查NVIDIA驱动和CUDA安装:
```bash
nvidia-smi
nvcc --version
```

重新安装PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 内存溢出(OOM)

调整配置:
```python
batch_size = 8           # 减半
context_size = 32        # 减半
num_workers = 2          # 减少
```

### 训练速度慢

检查瓶颈:
- 数据加载: 使用SSD，增加num_workers
- 计算能力: 使用更强GPU或增加GPU数量
- 网络通信: 检查分布式训练配置

## 引用

如使用本代码，请引用:
```bibtex
@software{plant_lm_2025,
  title={Plant Language Model Training System},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## 许可证

MIT License
