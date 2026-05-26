#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/train_latent.sh <dataset> [nfe]
# Example:
#   bash scripts/train_latent.sh lsun 4
#   bash scripts/train_latent.sh flux 4

dataset="${1:-lsun}"
nfe="${2:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "${dataset}" = "lsun" ]; then
    dataset="lsun_bedroom_ldm"
elif [ "${dataset}" = "flux" ]; then
    dataset="ms_coco"
fi

config_path="configs/${dataset}.yaml"

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
run_valid="${RUN_VALID:-False}"

train_port="${TRAIN_PORT:-11111}"
nproc_per_node="${NPROC_PER_NODE:-4}"

if [ "${dataset}" = "ms_coco" ]; then
    if [ -z "${DATADIR:-}" ]; then
        teacher_nfe="$((nfe + 2))"
        datadir="${ROOT_DIR}/logs/indis/data/ms_cocotime_uniform${teacher_nfe}"
    else
        datadir="${DATADIR}"
    fi
    torchrun --standalone --nproc_per_node="${nproc_per_node}" --master_port="${train_port}" \
        train.py --mnfe="${nfe}" --coslr --seed=0 --outdir="${outdir}" --config_path="${config_path}" \
        --run_valid="${run_valid}" --ema_decay_list_kimg "${ema_decay_list_kimg}" --datadir="${datadir}"
else
    torchrun --standalone --nproc_per_node="${nproc_per_node}" --master_port="${train_port}" \
        train.py --mnfe="${nfe}" --coslr --seed=0 --outdir="${outdir}" --config_path="${config_path}" \
        --run_valid="${run_valid}" --ema_decay_list_kimg "${ema_decay_list_kimg}"
fi
