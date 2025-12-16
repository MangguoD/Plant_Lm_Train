# Plant Language Model Training System

Production-ready training system for plant genome language models.

## Documentation

- **README_CN.md** - Complete technical documentation (Chinese)
- **data/README.md** - Data preparation instructions

## Quick Start

```bash
# Setup environment
bash setup_env.sh

# Prepare data (see data/README.md)
mkdir -p data/species_name
# Add FASTA files to data/species_name/

# Run training
bash run.sh
```

## Requirements

- Python 3.8+
- 16GB+ RAM
- 50GB+ disk space
- NVIDIA GPU (recommended)

## Configuration

Runtime parameters in `run.sh`:
- DATA_DIR
- WORLD_SIZE

Model hyperparameters in `plant_lm_train.py` Config class.

## Output

- Models: `dna_model/model_epoch_N.pt`
- Logs: `logs/training_TIMESTAMP.log`

## Verification

```bash
bash verify_setup.sh
```
