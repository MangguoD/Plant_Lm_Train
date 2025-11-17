#!/bin/bash
# Environment verification script

echo "=== Environment Verification ==="

# Check venv
echo "[1] Virtual environment"
if [ -d "venv" ]; then
    echo "OK: venv exists"
    source venv/bin/activate
else
    echo "FAIL: venv not found. Run: bash setup_env.sh"
    exit 1
fi

# Check Python
echo "[2] Python version"
python3 --version
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if [ $? -eq 0 ]; then
    echo "OK: Python >= 3.8"
else
    echo "FAIL: Python < 3.8"
fi

# Check packages
echo "[3] Package verification"
python3 << 'EOF'
packages = ['torch', 'Bio', 'numpy', 'tqdm', 'einops', 'wandb', 'psutil']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"OK: {pkg} {ver}")
    except:
        print(f"FAIL: {pkg}")
EOF

# Check CUDA
echo "[4] CUDA availability"
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"OK: CUDA available")
    print(f"  Version: {torch.version.cuda}")
    print(f"  GPUs: {torch.cuda.device_count()}")
else:
    print("INFO: CUDA not available (CPU mode)")
EOF

# Check files
echo "[5] Project files"
for file in plant_lm_train.py requirements.txt setup_env.sh run.sh; do
    if [ -f "$file" ]; then
        echo "OK: $file"
    else
        echo "FAIL: $file not found"
    fi
done

# Check data
echo "[6] Data directory"
if [ -d "data" ]; then
    echo "OK: data/ exists"
    count=$(find data -name "*.fna" -o -name "*.fa" -o -name "*.fasta" | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "OK: Found $count FASTA files"
    else
        echo "INFO: No FASTA files found"
    fi
else
    echo "INFO: data/ not found"
fi

# Check resources
echo "[7] System resources"
python3 << 'EOF'
import psutil
print(f"CPU cores: {psutil.cpu_count()}")
print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"Disk free: {psutil.disk_usage('.').free / 1024**3:.1f} GB")
EOF

echo "=== Verification Complete ==="
