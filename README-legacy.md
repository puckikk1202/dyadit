# Inference Script Standalone Package

This is a standalone package for running inference with `inference_dyadic.py`.

## Directory Structure

```
dyadit_code/
├── inference_dyadic.py    # Main inference script
├── tool.py                  # Utility functions for visualization and saving
├── configs/                 # Configuration files
│   └── dyadit_config.yaml
├── model/                   # Model definitions
│   ├── __init__.py
│   ├── dyadit.py
│   ├── attention.py
│   └── motion_generation/
├── datasets/                # Dataset loaders
│   ├── __init__.py
│   └── dyadic_loader.py
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── motion_utils.py
│   └── common.py
└── dyadic_dit_vq/          # VQ-VAE model
    ├── model/
    │   ├── __init__.py
    │   ├── vqvae.py
    │   └── vqvae_downsampled.py
    └── utils/
        └── motion_utils.py
```

## Usage

1. **Update configuration file**: Edit `configs/dyadit_config.yaml` to set correct paths:
   - `paths.data_root`: Path to your data directory
   - `vqvae.checkpoint_path`: Path to VQ-VAE checkpoint
   - `paths.device`: Device to use (e.g., "cuda" or "cuda:0")

2. **Run inference**:
```bash
python inference_dyadic.py \
    --config configs/dyadit_config.yaml \
    --checkpoint /path/to/your/checkpoint.ckpt \
    --batch_idx 0 \
    --num_batches 1 \
    --mode quality
```

## Required Arguments

- `--config`: Path to YAML configuration file
- `--checkpoint`: Path to model checkpoint file

## Optional Arguments

- `--batch_idx`: Starting batch index (default: 0)
- `--num_batches`: Number of batches to process (default: 1)
- `--mode`: Inference mode - `quality`, `relat`, `ps`, `uncond`, `random`, `relat_random`, `ps_random` (default: quality)
- `--output_prefix`: Output directory prefix (default: inference_outputs)
- `--win_length`: Sliding window length (default: 384)
- `--stride`: Stride between windows (default: 384)
- `--guidance_scale`: CFG guidance scale (default: 4.0)
- `--inference_steps`: Number of diffusion steps (default: 50)
- `--save_visuals`: Save visualization images

## Dependencies

Make sure you have the following packages installed:
- torch
- torchvision
- numpy
- matplotlib
- yaml
- transformers (for Wav2Vec2)
- pytorch-lightning
- diffusers
- librosa
- soundfile
- pandas
- einops

## Notes

- The script imports from local modules, so make sure to run it from this directory or add it to your PYTHONPATH
- All paths in the config file should be absolute paths or relative to where you run the script
- The VQ-VAE checkpoint path must be valid and accessible
