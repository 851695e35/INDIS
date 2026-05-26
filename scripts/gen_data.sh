#!/bin/bash
set -euo pipefail

# Single-entrypoint data generation for public release.
# Usage:
#   ./scripts/gen_data.sh [dataset] [num_steps]
# Example:
#   ./scripts/gen_data.sh cifar10 30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

dataset="${1:-imagenet64}" # cifar10, ffhq, afhqv2, imagenet64, lsun_bedroom_ldm, ms_coco
num_steps="${2:-30}"

guidance_type="none"
guidance_rate="0.0"
solver="ipndm" # ipndm uni_pc
batch="${BATCH:-32}"
schedule_type="logsnr"
schedule_rho="7"
noise_schedule="VE"

if [ "${dataset}" = "lsun_bedroom_ldm" ] || [ "${dataset}" = "ms_coco" ]; then
    schedule_type="time_uniform"
    schedule_rho="1"
    noise_schedule="VP"
fi

if [ "${dataset}" = "ms_coco" ]; then
    guidance_type="cfg"
    guidance_rate="4.0"
    solver="ipndm_flux"
    noise_schedule="OT"
fi

max_order="3"
outdir="${OUTDIR:-${ROOT_DIR}/logs/indis/data/${dataset}${schedule_type}${num_steps}}"
prompt_path="${PROMPT_PATH:-${APT_ROOT}/datasets/MS-COCO_val2014_30k_captions.csv}"
train_num="${TRAIN_NUM:-10000}"
valid_num="${VALID_NUM:-1000}"
device="${DEVICE:-cuda:0}"
pretrained_root="${PRETRAINED_ROOT:-${APT_ROOT}/pretrained/edm}"

declare -A model_dict
model_dict["cifar10"]="edm-cifar10-32x32-uncond-vp.pkl"
model_dict["ffhq"]="edm-ffhq-64x64-uncond-vp.pkl"
model_dict["afhqv2"]="edm-afhqv2-64x64-uncond-vp.pkl"
model_dict["imagenet64"]="edm-imagenet-64x64-cond-adm.pkl"
model_dict["lsun_bedroom_ldm"]="lsun_bedroom_ldm.ckpt"
model_dict["ms_coco"]="flux-dev"

if [ -z "${model_dict[$dataset]+x}" ]; then
    echo "Unsupported dataset: ${dataset}"
    exit 1
fi

pretrained_path="${pretrained_root}/${model_dict[$dataset]}"

python sample_tea.py \
    --batch="${batch}" \
    --outdir="${outdir}" \
    --data_num="${train_num}" \
    --validation_num="${valid_num}" \
    --schedule_type="${schedule_type}" \
    --schedule_rho="${schedule_rho}" \
    --max_order="${max_order}" \
    --num_steps="${num_steps}" \
    --noise_schedule="${noise_schedule}" \
    --prompt_path="${prompt_path}" \
    --guidance_type="${guidance_type}" \
    --guidance_rate="${guidance_rate}" \
    --model_path="${pretrained_path}" \
    --dataset_name="${dataset}" \
    --device="${device}" \
    --solver="${solver}" \
    --afs=False
