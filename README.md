# DyaDiT

Training, motion generation, and two-person SMPL-H rendering with synchronized audio.

## Installation

```bash
git clone https://github.com/puckikk1202/dyadit.git
cd dyadit
pip install -r requirements.txt
pip install -r requirements-render.txt
```

Install `librosa` if your dataset uses WAV or MP3 files:

```bash
pip install librosa
```

You will need a pretrained motion VQ-VAE checkpoint and SMPL-H model assets. The audio encoder uses `facebook/wav2vec2-base-960h`; a local model directory can be set through `model.audio_model_name` for offline use.

## Data

Prepare separate training and validation directories:

```text
data_root/
├── motion_6d_slim/*.npy
├── audio_slim/*.npy
├── csv/relationships.csv
├── csv/participants.csv
└── motion_6d_normalization.npz
```

Motion files can also be stored in `motion_6d/*.npz` or `motion/*.npz`, and audio in `audio_npz/*.npz` or `audio/*.wav`. Motion and audio files use matching names such as `Vxx_Sxxxx_Ixxxxxxxx_Pxxxx`, with two participants per conversation.

Compute training normalization statistics and prepare NumPy files:

```bash
python preprocess.py --data-root /path/to/train --slim
```

Use the training normalization statistics for both training and validation.

## Configuration

Set these fields in [`configs/core.yaml`](configs/core.yaml):

```yaml
paths:
  data_root: /path/to/train
  val_data_root: /path/to/validation
  normalization_stats: /path/to/train/motion_6d_normalization.npz
  output_dir: /path/to/run

vqvae:
  checkpoint_path: /path/to/net_best.pth

visualization:
  model_path: /path/to/smplh_models

logger:
  type: wandb
  entity: your-wandb-entity
  project: dyadit
```

Relative paths are resolved from the configuration file's directory. Override individual settings with `--set key=value`.

## Training

Sign in to W&B, then start training on one GPU:

```bash
wandb login
python train.py --config configs/core.yaml \
  --set trainer.devices=1 --set data.batch_size=64
```

For four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 \
  train.py --config configs/core.yaml
```

Add `--resume /path/to/last.ckpt` to resume training. Use `--weights /path/to/model.ckpt` to initialize a new run from model weights.

The default configuration uses 96-frame training clips at 30 fps. Outputs are saved under `paths.output_dir`, including `config.yaml`, `checkpoints/`, W&B logs, metrics, and rendered videos.

Every epoch logs motion-FID to `val/fid_codec` and a curve to `fid/curve`; CSV and PNG files are saved in `metrics/fid_codec/`. Every five epochs, ten 600-frame samples are rendered to `visualization/samples`. Validation conversations must be long enough for these previews.

To disable video previews:

```bash
python train.py --config configs/core.yaml \
  --set visualization.enabled=false
```

## Inference

Use the configuration saved with your training run:

```bash
python inference.py \
  --config /path/to/run/config.yaml \
  --checkpoint /path/to/run/checkpoints/last.ckpt \
  --data-root /path/to/test \
  --index 0 --num-samples 1 \
  --pose-length 600 \
  --window-frames 96 --stride-frames 80 \
  --inference-steps 50 \
  --output-dir inference_outputs
```

Sequence length, window size, and stride are measured in motion frames and must be divisible by the codec downsampling factor (default: 4). The target-motion prefix is used by default; add `--no-reference-prefix` to omit it.

Each sample directory contains SMPL-H motion files, both audio tracks, latent features, and generation metadata.

## Rendering

```bash
python render.py \
  --sample-dir inference_outputs/sample_00000 \
  --model-path /path/to/smplh_models \
  --backend software
```

The two people face each other with their lower bodies held in a neutral pose.

| Person | Motion | Audio channel |
| --- | --- | --- |
| Blue, left | Conditioning person B | Left |
| Orange, right | Generated person A | Right |

The renderer saves `conversation.mp4`, a preview image, and rendering settings. Use `--audio-layout mix` for mixed audio or `--no-audio-normalize` to keep the source volume.

To render directly after inference, append:

```bash
--render --smplh-model /path/to/smplh_models --render-backend software
```

## TSUBAME submission

Submit a four-GPU job from the repository root:

```bash
mkdir -p outputs/qsub_logs
qsub qsub_4gpu.sh
```

The launcher uses the `tga-iil-general` billing group. Set `DYADIT_PYTHON`, `DYADIT_TRAIN_ROOT`, `DYADIT_VAL_ROOT`, `DYADIT_CODEC`, and `DYADIT_AUDIO_MODEL` for your environment, and pass overrides with `qsub -v`. W&B destinations can be set with `WANDB_ENTITY` and `WANDB_PROJECT`.

## Tests

```bash
pip install pytest
WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q
```
