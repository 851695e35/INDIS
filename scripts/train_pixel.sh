#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/train_pixel.sh <dataset> [nfe]
# Example:
#   bash scripts/train_pixel.sh cifar10 4

dataset="${1:-cifar10}"
nfe="${2:-4}"
shift $(( $# >= 2 ? 2 : $# ))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

config_path="${CONFIG_PATH:-configs/${dataset}.yaml}"
if [ ! -f "${config_path}" ]; then
    echo "Config not found: ${config_path}"
    echo "Supported pixel datasets: cifar10, afhqv2, ffhq, imagenet64"
    exit 1
fi

name="${NAME-}"
tag="${TAG:-}"
if [ -n "${OUTDIR:-}" ]; then
    outdir="${OUTDIR}"
elif [ -n "${name}" ] || [ -n "${tag}" ]; then
    outdir="${ROOT_DIR}/logs/indis/${name}${dataset}${tag}"
else
    outdir="${ROOT_DIR}/logs/indis"
fi
ema_decay_list_kimg="${EMA_DECAY_LIST_KIMG:-0,2}"
run_valid="${RUN_VALID:-True}"
extra_args=()

if [ -n "${DETERMINISTIC:-}" ]; then
    : "${CUBLAS_WORKSPACE_CONFIG:=:4096:8}"
    export CUBLAS_WORKSPACE_CONFIG
    : "${PYTHONHASHSEED:=0}"
    export PYTHONHASHSEED
fi

if [ -n "${DATADIR:-}" ]; then
    extra_args+=(--datadir="${DATADIR}")
fi
if [ -n "${BATCH:-}" ]; then
    extra_args+=(--batch="${BATCH}")
fi
if [ -n "${BATCH_GPU:-}" ]; then
    extra_args+=(--batch-gpu="${BATCH_GPU}")
fi
if [ -n "${TRAIN_NUM:-}" ]; then
    extra_args+=(--train_num="${TRAIN_NUM}")
fi
if [ -n "${VALID_NUM:-}" ]; then
    extra_args+=(--valid_num="${VALID_NUM}")
fi
if [ -n "${NUM_WORKERS:-}" ]; then
    extra_args+=(--num_workers="${NUM_WORKERS}")
fi
if [ -n "${BENCH:-}" ]; then
    extra_args+=(--bench="${BENCH}")
fi
if [ -n "${DETERMINISTIC:-}" ]; then
    extra_args+=(--deterministic="${DETERMINISTIC}")
fi
if [ -n "${MAX_GRAD_NORM:-}" ]; then
    extra_args+=(--max_grad_norm="${MAX_GRAD_NORM}")
fi
if [ -n "${LR_PARAM:-}" ]; then
    extra_args+=(--lr_param="${LR_PARAM}")
fi
if [ -n "${LR_NET:-}" ]; then
    extra_args+=(--lr_net="${LR_NET}")
fi
if [ -n "${WEIGHT_DECAY_PARAM:-}" ]; then
    extra_args+=(--weight_decay_param="${WEIGHT_DECAY_PARAM}")
fi
if [ -n "${WEIGHT_DECAY_NET:-}" ]; then
    extra_args+=(--weight_decay_net="${WEIGHT_DECAY_NET}")
fi
if [ -n "${WARMUP_START_LR:-}" ]; then
    extra_args+=(--warmup_start_lr="${WARMUP_START_LR}")
fi
if [ -n "${ETA_MIN:-}" ]; then
    extra_args+=(--eta_min="${ETA_MIN}")
fi
if [ -n "${WARMUP_RATIO:-}" ]; then
    extra_args+=(--warmup_ratio="${WARMUP_RATIO}")
fi
if [ -n "${LR_SCHEDULER_TYPE:-}" ]; then
    extra_args+=(--lr_scheduler_type="${LR_SCHEDULER_TYPE}")
fi
if [ -n "${PLATEAU_FACTOR:-}" ]; then
    extra_args+=(--plateau_factor="${PLATEAU_FACTOR}")
fi
if [ -n "${PLATEAU_PATIENCE:-}" ]; then
    extra_args+=(--plateau_patience="${PLATEAU_PATIENCE}")
fi
if [ -n "${PLATEAU_THRESHOLD:-}" ]; then
    extra_args+=(--plateau_threshold="${PLATEAU_THRESHOLD}")
fi
if [ -n "${PLATEAU_MIN_LR:-}" ]; then
    extra_args+=(--plateau_min_lr="${PLATEAU_MIN_LR}")
fi
if [ -n "${EARLY_STOP_PATIENCE:-}" ]; then
    extra_args+=(--early_stop_patience="${EARLY_STOP_PATIENCE}")
fi
if [ -n "${EARLY_STOP_MIN_DELTA:-}" ]; then
    extra_args+=(--early_stop_min_delta="${EARLY_STOP_MIN_DELTA}")
fi
if [ -n "${INIT_PREDICTOR_PATH:-}" ]; then
    extra_args+=(--init_predictor_path="${INIT_PREDICTOR_PATH}")
fi

# Single-GPU training
python train.py --mnfe="${nfe}" --coslr --seed=0 --outdir="${outdir}" --config_path="${config_path}" \
    --run_valid="${run_valid}" --ema_decay_list_kimg "${ema_decay_list_kimg}" "${extra_args[@]}" "$@"
