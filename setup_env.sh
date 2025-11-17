#!/bin/bash
# Environment setup script
set -e

VENV_DIR="venv"

echo "[1/5] Checking Python version..."
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8+ required"
    exit 1
fi
python3 --version

echo "[2/5] Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR"

echo "[3/5] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[4/5] Installing dependencies..."
pip install --upgrade pip setuptools wheel -q
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found"
    exit 1
fi

echo "[5/5] Verifying installation..."
python3 << 'EOF'
import sys
packages = ['torch', 'Bio', 'numpy', 'tqdm', 'einops', 'wandb', 'psutil']
all_ok = True
for pkg in packages:
    try:
        __import__(pkg)
        print(f"OK: {pkg}")
    except ImportError:
        print(f"FAIL: {pkg}")
        all_ok = False
sys.exit(0 if all_ok else 1)
EOF

if [ $? -eq 0 ]; then
    echo "Setup complete."
    python3 -c "import torch; print('CUDA:', 'available' if torch.cuda.is_available() else 'not available')"
else
    echo "Setup failed"
    exit 1
fi
