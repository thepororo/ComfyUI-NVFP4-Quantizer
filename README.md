# ComfyUI Universal FP8 / NVFP4 Quantizer

[English](README.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

A self-contained ComfyUI output node for mixed FP8/NVFP4 quantization of
diffusion-model `.safetensors` files.

## Supported architectures

- SDXL
- Wan 2.2
- Qwen Image and Qwen Image Edit
- Z-Image
- FLUX and FLUX.2
- Generic transformer fallback

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thepororo/ComfyUI-NVFP4-Quantizer.git
```

Restart ComfyUI. The node appears under `model/quantization` as
`Universal FP8 / NVFP4 Quantizer`.

## Usage

1. Select a model from ComfyUI's `models/diffusion_models` list.
2. Use `architecture=auto`, `preset=balanced`, and `estimate_only=true`.
3. Review the detected architecture and estimated output size in `log`.
4. Disable `estimate_only` to create the quantized model.

The output is written beside the selected source model. Leave
`output_filename` empty to use `<source_name>_nvfp4.safetensors`.

## Presets

- `balanced`: attention FP8, FFN NVFP4, other large Linear layers FP8.
- `quality`: attention FP8, FFN NVFP4, miscellaneous Linear layers BF16.
- `aggressive`: attention and FFN NVFP4.
- `fp8_all`: large eligible Linear layers FP8.
- `custom`: use the individual format controls.

Sensitive layers, small matrices, convolution weights, norms, and biases are
protected according to the selected settings. SDXL convolution weights cannot
use NVFP4 because this implementation supports 2D Linear weights only.

## Requirements and warnings

- NVFP4 conversion requires CUDA and a ComfyUI/comfy-kitchen build containing
  `TensorCoreNVFP4Layout`.
- Start from an original BF16/FP16 model, not an already quantized file.
- Quantization is lossy. Keep the source model.
- Test model quality before production use.

## Development note

Designed and maintained by [thepororo](https://github.com/thepororo) with
implementation assistance from OpenAI Codex. AI-assisted changes are reviewed
and validated locally before publication.
