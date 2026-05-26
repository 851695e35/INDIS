#!/bin/bash
set -euo pipefail

# Fill these placeholders before running.
# dataset supports: lsun (lsun_bedroom_ldm), flux (ms_coco)
dataset="..."
config_path="configs/....yaml"

# Hyperparameters / paths (fill in)
nproc_per_node="4"
sample_port="22222"
nfe="..."
snapshot_kimg="..."
predicted_path="..."
sampling_batch="..."
seeds="..."
output_path="..."
model_path="..."
noise_schedule="..."

# Flux (ms_coco) only; keep empty for lsun
prompt_path="..."

# Optional: parse from config instead of manually filling
# model_path=$(grep -oP 'model_path:\s*\K[^ ]+' "${config_path}")
# noise_schedule=$(grep -oP 'noise_schedule:\s*\K[^ ]+' "${config_path}")
# sampling_batch=$(grep -oP 'sampling_batch:\s*\K[^ ]+' "${config_path}")

mkdir -p "${output_path}"

if [ "${dataset}" = "flux" ] || [ "${dataset}" = "ms_coco" ]; then
    torchrun --standalone --nproc_per_node="${nproc_per_node}" --master_port="${sample_port}" \
        sample.py --predictor_path="${predicted_path}" --batch="${sampling_batch}" --seeds="${seeds}" \
        --prompt_path="${prompt_path}" --outdir="${output_path}" --model_path="${model_path}" --noise_schedule="${noise_schedule}"
else
    torchrun --standalone --nproc_per_node="${nproc_per_node}" --master_port="${sample_port}" \
        sample.py --predictor_path="${predicted_path}" --batch="${sampling_batch}" --seeds="${seeds}" \
        --outdir="${output_path}" --model_path="${model_path}" --noise_schedule="${noise_schedule}"
fi
