#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -A tga-iil-general
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N dyadit_core_4g
#$ -j y
#$ -o outputs/qsub_logs

# TSUBAME: one full node supplies four GPUs. Submit from this directory:
#   mkdir -p outputs/qsub_logs
#   qsub qsub_4gpu.sh
# Credentials are read from the existing ~/.netrc, never copied into this script.
set -eo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.8.0
module load cudnn/9.8.0
set -u

task_core="${DYADIT_CORE_DIR:-${SGE_O_WORKDIR:?Submit from the repository root with qsub}}"
task_python="${DYADIT_PYTHON:-/gs/fs/tga-koike-crescent/miniconda3/envs/mogen_py310/bin/python}"
task_train="${DYADIT_TRAIN_ROOT:-/gs/bs/tga-koike-crescent/datasets/seamless_dataset/train}"
task_validation="${DYADIT_VAL_ROOT:-/gs/fs/tga-koike-crescent/peng/dyadic_dit/data}"
task_codec="${DYADIT_CODEC:-/gs/fs/tga-koike-crescent/peng/dyadic_dit/dyadic_dit_vq/ckpt/net_best.pth}"
task_audio="${DYADIT_AUDIO_MODEL:-/gs/fs/tga-koike-crescent/.cache-peng/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156}"
task_output="${DYADIT_OUTPUT_DIR:-${task_core}/outputs/cluster/job_${JOB_ID:?Submit this script through qsub}}"
task_run_name="${DYADIT_RUN_NAME:-dyadit_core_4gpu_${JOB_ID}}"

cd "$task_core"
mkdir -p "$task_output/cache/matplotlib" "$task_output/cache/wandb" \
    "$task_output/cache/wandb-config" "$task_output/cache/wandb-data" \
    "$task_output/cache/huggingface"
export PATH="$(dirname -- "$task_python"):$PATH"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$task_output/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MPLCONFIGDIR="$task_output/cache/matplotlib"
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-ohicarip}"
export WANDB_PROJECT="${WANDB_PROJECT:-wan_dit_audio2pose}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-dyadit-core-${JOB_ID}}"
export WANDB_RESUME=allow
export WANDB_DIR="$task_output"
export WANDB_CACHE_DIR="$task_output/cache/wandb"
export WANDB_CONFIG_DIR="$task_output/cache/wandb-config"
export WANDB_DATA_DIR="$task_output/cache/wandb-data"

printf 'Job: %s | billing group: tga-iil-general | host: %s\n' "$JOB_ID" "${HOSTNAME:-unknown}"
printf 'Output: %s\nW&B: https://wandb.ai/%s/%s/runs/%s\n' \
    "$task_output" "$WANDB_ENTITY" "$WANDB_PROJECT" "$WANDB_RUN_ID"
"$task_python" -u - <<'PY'
import importlib
import netrc
import os

import torch

for name in ("wandb", "smplx", "scipy", "cv2", "imageio", "imageio_ffmpeg", "soundfile"):
    importlib.import_module(name)
if not os.environ.get("WANDB_API_KEY"):
    auth = netrc.netrc().authenticators("api.wandb.ai")
    if not auth or not auth[2]:
        raise RuntimeError("No W&B credentials; configure wandb login before submission")
if torch.cuda.device_count() != 4:
    raise RuntimeError(f"Expected four allocated GPUs, got {torch.cuda.device_count()}")
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, flush=True)
for index in range(4):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
PY

exec "$task_python" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 \
    train.py --config configs/core.yaml \
    --set "exp_name=${task_run_name}" \
    --set "paths.data_root=${task_train}" \
    --set "paths.val_data_root=${task_validation}" \
    --set "paths.normalization_stats=${task_train}/motion_6d_normalization.npz" \
    --set "paths.output_dir=${task_output}" \
    --set "vqvae.checkpoint_path=${task_codec}" \
    --set "model.audio_model_name=${task_audio}" \
    --set trainer.devices=4 \
    --set logger.type=wandb \
    --set "logger.entity=${WANDB_ENTITY}" \
    --set "logger.project=${WANDB_PROJECT}" \
    --set fid.enabled=true \
    --set fid.every_n_epochs=1 \
    --set visualization.enabled=true \
    --set visualization.every_n_epochs=5 \
    --set visualization.num_samples=10 \
    --set visualization.pose_length=600 \
    --set visualization.window_frames=96 \
    --set visualization.stride_frames=80 \
    --set visualization.render.audio_layout=stereo \
    "$@"
