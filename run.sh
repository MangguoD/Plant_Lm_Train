#!/bin/bash
# Training launcher
set -e

# Configuration
DATA_DIR="./data"
WORLD_SIZE=2
export WANDB_MODE="online"
export WANDB_PROJECT="plant-genome-lm"
export WANDB_ENTITY=""

echo "[1/4] Checking environment..."
if [ ! -d "venv" ]; then
    echo "Error: venv not found. Run: bash setup_env.sh"
    exit 1
fi
source venv/bin/activate

echo "[2/4] Checking data directory..."
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: data directory not found: $DATA_DIR"
    exit 1
fi
fasta_count=$(find "$DATA_DIR" -type f \( -name "*.fna" -o -name "*.fa" -o -name "*.fasta" \) | wc -l)
if [ "$fasta_count" -eq 0 ]; then
    echo "Error: no FASTA files found in $DATA_DIR"
    exit 1
fi
echo "Found $fasta_count FASTA files"

echo "[3/4] Checking GPU..."
gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$gpu_count" -eq 0 ]; then
    echo "Warning: No GPU detected. Training will be slow."
    WORLD_SIZE=1
else
    echo "Detected $gpu_count GPUs"
    if [ "$WORLD_SIZE" -gt "$gpu_count" ]; then
        WORLD_SIZE=$gpu_count
    fi
    echo "Using $WORLD_SIZE GPUs"
fi

echo "[4/4] Preparing output directories..."
OUTPUT_DIR="dna_model"
LOG_DIR="logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "Starting training..."
echo "Data: $DATA_DIR"
echo "GPUs: $WORLD_SIZE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python3 plant_lm_train.py \
    --world_size "$WORLD_SIZE" \
    --data_dir "$DATA_DIR" \
    2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    echo "Models: $OUTPUT_DIR/"
    echo "Log: $LOG_FILE"
else
    echo "Training failed. Check log: $LOG_FILE"
    exit 1
fi
