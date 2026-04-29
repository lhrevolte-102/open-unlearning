# Ascend NPU Notes

This repository defaults to a CUDA-oriented setup:

- `flash_attention_2` is enabled in multiple model configs.
- `paged_adamw_32bit` is used as the default optimizer.
- the baseline launch scripts assume `CUDA_VISIBLE_DEVICES` and a DeepSpeed config.

On Ascend NPU, use the Ascend-specific path added in this repository instead.

## Environment

Use an environment that already has a working `torch` + `torch_npu` pair for your local CANN installation.

Then install the project dependencies that do not force CUDA-specific packages:

```bash
pip install -r requirements-ascend.txt
```

`requirements-ascend.txt` includes `einops` because the `transformers` NPU flash-attention integration imports it even when this repository overrides model attention backends to `eager`.

This file intentionally excludes:

- `torch`: use the version paired with your local `torch_npu`
- `bitsandbytes`: this repo uses `paged_adamw_32bit`, but Ascend only documents NF4 quant/dequant support for QLoRA-style flows
- `deepspeed`: Hugging Face + Ascend documents DeepSpeed support as experimental, so the baseline Ascend path here uses `accelerate` multi-NPU instead

## Entry Points

`src/train.py` and `src/eval.py` now try to import `torch_npu` at process start. This follows the Ascend guidance for `transformers`/`accelerate` style entry scripts.

## Launching TOFU on Ascend

Use the dedicated script:

```bash
bash scripts/tofu_unlearn_ascend.sh
```

It changes the default behavior in three ways:

- uses `configs/accelerate/ascend_multi_npu.yaml`
- switches `attn_implementation` to `eager`
- switches the optimizer to `adamw_torch`

## Device Selection

Do not use the original `CUDA_VISIBLE_DEVICES=...` prefixes from the upstream scripts.

Instead, configure Ascend visibility outside the script when needed, for example:

```bash
export ASCEND_VISIBLE_DEVICES=14,15
```
