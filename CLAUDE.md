# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**str(mouse)** decodes cursor movement into characters (a-z, A-Z, space) by transforming mouse velocity data into images classified by a lightweight CNN. The system achieves 400+ chars/s on M2 Pro MacBook with 92.9% accuracy.

## Common Commands

### Development
```bash
# Run in virtual environment using uv
uv run -m <module_path>

# Try real-time decoding (main demo)
uv run -m src.decode.real_time --ckpt best_finetune

# Collect training data
uv run -m src.data.collect_data

# Test model on sample velocity data
uv run main.py --ckpt <ckpt_name>
```

### Training Pipeline
```bash
# 1. Pretrain on EMNIST (27 classes: a-z + space)
uv run -m src.ml.emnist_pretrain

# 2. Finetune on collected mouse data (53 classes: a-z, A-Z, space)
uv run -m src.ml.finetune --ckpt <pretrain_ckpt_name>

# 3. Train LSTM model (experimental, uses raw velocity sequences)
uv run -m src.ml.pretrain --model lstm
```

### API Server
```bash
# Start FastAPI server for browser-based data collection
uv run -m src.api.main
# Then navigate to http://localhost:8000
```

## Architecture

### Two-Stage Training Approach
1. **Pretraining**: StrokeNet CNN trained on EMNIST lowercase letters (27 classes)
2. **Finetuning**: Replace classifier head with 53-class head, freeze early layers (block1, block2), train only block3 + classifier on user's mouse data

### Data Flow: Mouse → Character
1. **Collection** (`src/data/collect_data.py`): Record mouse velocities until 500ms pause
2. **Preprocessing** (`src/data/utils.py:build_img()`):
   - Integrate velocities → positions
   - Render as 28x28 grayscale image
3. **Inference** (`src/ml/architectures/cnn.py:StrokeNet`): CNN classifies image → character

### Model Architectures
- **StrokeNet** (`src/ml/architectures/cnn.py`): Primary CNN model
  - Input: 1x28x28 grayscale images
  - 3 conv blocks (16→32→64→128 channels) + Global Average Pooling
  - 0.3M parameters
  - Pretrain: 27 classes (lowercase + space)
  - Finetune: 53 classes (full charset)

- **StrokeLSTM** (`src/ml/architectures/lstm.py`): Experimental time-series model
  - Input: Variable-length (T, 2) velocity sequences (PackedSequence)
  - No pretrain/finetune separation (lacks abundant pretraining dataset)

### Character Encoding
Character → Index mapping (`src/data/utils.py`):
- Lowercase a-z: 0-25
- Uppercase A-Z: 26-51
- Space: 52

### Checkpoint Structure
```
checkpoints/
├── emnist_pretrain/     # 27-class pretrained models
│   └── <wandb_run_name>/
│       └── best.ckpt
└── mouse_finetune/      # 53-class finetuned models
    └── <wandb_run_name>/
        └── best.ckpt
```

### Data Storage
- **Raw mouse data**: `raw_mouse_data/<char_folder>/trial_DD-HH-MM-SS-mmm.csv`
  - Char folders: `a-z`, `uA-uZ` (uppercase prefix for case-insensitive filesystems), `space`
- **AWS S3**: Public bucket `s3://hectorastrom-str-mouse/raw_mouse_data`
  - Download with: `aws s3 sync s3://hectorastrom-str-mouse/raw_mouse_data ./raw_mouse_data`

### Key Modules
- **src/data/**: Data collection, preprocessing, and utilities
  - `collect_data.py`: Interactive data collection interface
  - `utils.py`: Core utilities (build_img, char_map, GlobalInputManager)
- **src/ml/**: Training scripts and model architectures
  - `architectures/cnn.py`: StrokeNet implementation
  - `architectures/lstm.py`: StrokeLSTM implementation
  - `emnist_pretrain.py`: Pretrain on EMNIST dataset
  - `pretrain.py`: Pretrain on user's collected data (for LSTM)
  - `finetune.py`: Finetune pretrained CNN on user data
- **src/decode/**: Inference and decoding
  - `real_time.py`: Live decoding demo
  - `decipher.py`: Batch processing utilities
- **src/api/**: FastAPI server for browser-based data collection
- **src/visualize/**: Visualization utilities

### Training Configuration
- **Pretraining**: AdamW (lr=1e-3, weight_decay=1e-4), all parameters
- **Finetuning**: Adam (lr=1e-4), only block3 + classifier parameters
- Both use PyTorch Lightning with WandB logging
- Early stopping available via `--early-stop` flag

## Project Context
- Uses `uv` for dependency management (not pip)
- Python 3.11+ required
- Uses PyTorch Lightning for training
- WandB for experiment tracking
- Mouse data collected via `pynput` library on macOS
