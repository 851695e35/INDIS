# INDIS

This release contains the content for INDIS:

1. Generate data with `scripts/gen_data.sh`.
2. Train with `scripts/train_pixel.sh` / `scripts/train_latent.sh`.
3. Sample with `scripts/sample_pixel.sh` / `scripts/sample_latent.sh`.

## Data Generation

Run:

```bash
bash scripts/gen_data.sh <dataset> <num_steps>
```

Examples:

```bash
bash scripts/gen_data.sh cifar10 30
bash scripts/gen_data.sh ms_coco 10
```

Supported datasets:
- `cifar10`
- `ffhq`
- `afhqv2`
- `imagenet64`
- `lsun_bedroom_ldm`
- `ms_coco`

## Train (Single NFE)

### Pixel-space datasets

Run:

```bash
bash scripts/train_pixel.sh <dataset> <nfe>
```

Examples:

```bash
bash scripts/train_pixel.sh cifar10 4
bash scripts/train_pixel.sh afhqv2 4
bash scripts/train_pixel.sh ffhq 4
bash scripts/train_pixel.sh imagenet64 4
```

Supported pixel datasets:
- `cifar10`
- `afhqv2`
- `ffhq`
- `imagenet64`

### Latent-space datasets

Run:

```bash
bash scripts/train_latent.sh <dataset> <nfe>
```

Examples:

```bash
bash scripts/train_latent.sh lsun 4
bash scripts/train_latent.sh flux 4
```

Supported latent aliases:
- `lsun` (mapped to `lsun_bedroom_ldm`)
- `flux` (mapped to `ms_coco`)

## Sampling

Sampling scripts are template entrypoints and require filling placeholders first.

### Pixel sampling

Edit and run:

```bash
bash scripts/sample_pixel.sh
```

Required fields in the script:
- `predicted_path`
- `sampling_batch`
- `seeds`
- `output_path`
- `model_path`
- `noise_schedule`

### Latent sampling

Edit and run:

```bash
bash scripts/sample_latent.sh
```

For flux (`ms_coco`), also set:
- `prompt_path`

For reference of the predictor ckpt: https://huggingface.co/Carbon787/INDIS/tree/main.

## Contact

If you have any questions or suggestions, feel free to contact at liangyuy001@gmail.com.
