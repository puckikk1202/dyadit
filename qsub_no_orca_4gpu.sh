#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -A tga-iil-general
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N dyadit_no_orca
#$ -j y
#$ -o outputs/qsub_logs

# A separate experiment using the shared training launcher and core.yaml.
# Submit without --resume/--weights for a fresh DiT ablation. The pretrained
# codec and speech encoder are the same as in the full model.
set -euo pipefail

task_core="${DYADIT_CORE_DIR:-${SGE_O_WORKDIR:?Submit from the repository root with qsub}}"
task_id="${JOB_ID:?Submit this script through qsub}"
export DYADIT_OUTPUT_DIR="${task_core}/outputs/cluster/no_orca_job_${task_id}"
export DYADIT_RUN_NAME="dyadit_no_orca_4gpu_${task_id}"
export WANDB_RUN_ID="dyadit-no-orca-${task_id}"

printf 'Ablation: model.orthogonal_fusion=false | W&B run: %s\n' "$WANDB_RUN_ID"
exec /bin/bash "${task_core}/qsub_4gpu.sh" "$@" \
    --set model.orthogonal_fusion=false
