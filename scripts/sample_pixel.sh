#!/bin/bash
set -euo pipefail

# Fill these placeholders before running.
dataset="..."
config_path="configs/....yaml"

# Hyperparameters / paths (fill in)
nfe="..."
snapshot_kimg="..."
predicted_path="..."
sampling_batch="..."
seeds="..."
output_path="..."
model_path="..."
noise_schedule="..."

# Optional: parse from config instead of manually filling
# model_path=$(grep -oP 'model_path:\s*\K[^ ]+' "${config_path}")
# noise_schedule=$(grep -oP 'noise_schedule:\s*\K[^ ]+' "${config_path}")
# sampling_batch=$(grep -oP 'sampling_batch:\s*\K[^ ]+' "${config_path}")

mkdir -p "${output_path}"

python sample.py --predictor_path="${predicted_path}" --batch="${sampling_batch}" --seeds="${seeds}" \
    --outdir="${output_path}" --model_path="${model_path}" --noise_schedule="${noise_schedule}"
