# ComfyUI Universal FP8 / NVFP4 Quantizer

[English](README.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-2b2b2b)](https://www.comfy.org/)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors)](https://github.com/sponsors/thepororo)

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

The package now provides separate nodes for SDXL, Wan 2.2, Qwen Image/Edit,
Z-Image, FLUX, and FLUX.2. Each node exposes its architecture's major layer
groups independently.

- `conservative_nvfp4` (default): keeps sensitive and input/output layers in
  their source dtype, uses safe FP8 weight-only storage for attention and
  reduction paths, and applies NVFP4 only to large MLP/expansion paths.
- `conservative_fp8`: keeps sensitive and input/output layers in their source
  dtype and stores eligible major Linear weights as FP8 weight-only.
- `custom`: enables all architecture-specific layer selectors.

FP8 layers never use an implicit activation scale: matrix multiplication stays
in the model compute dtype. NVFP4 tensor scales are forced positive, validated
as finite, and paired with the native per-block FP8 scales. Already-quantized
inputs are rejected.

- `balanced`: attention FP8, FFN NVFP4, other large Linear layers FP8.
- `quality`: attention FP8, FFN NVFP4, miscellaneous Linear layers BF16.
- `aggressive`: attention and FFN NVFP4.
- `qwen_edit_14_5gb`: Qwen Image/Edit 2511 profile targeting about 14.5 GB;
  attention, FFN, and other large Linear layers use NVFP4, while `img_mod` and
  `txt_mod` use FP8 and input/output embeddings remain BF16.
- `qwen_edit_under_14gb`: more aggressive Qwen Image/Edit 2511 profile targeting
  about 13.8 GB. It keeps modulation in the first and last 23 blocks at FP8 and
  converts modulation in middle blocks 23–36 to NVFP4.
- `qwen_edit_perceptual_hybrid`: quality-first Qwen Image/Edit 2511 profile.
  Attention and MLP reduction projections use FP8 weight-only execution, MLP
  expansion projections use NVFP4, modulation uses FP8 weight-only execution,
  and input/output, norms, and biases remain BF16. It prioritizes style,
  surface texture, and detail.
- `fp8_all`: large eligible Linear layers FP8.
- `custom`: use the individual format controls.

Sensitive layers, small matrices, convolution weights, norms, and biases are
protected according to the selected settings. SDXL convolution weights cannot
use NVFP4 because this implementation supports 2D Linear weights only.

FP8 layers are stored with per-tensor weight scaling and marked for
full-precision matrix multiplication. This keeps activations in the model's
compute dtype because this offline converter does not perform activation
calibration or generate `input_scale` tensors. NVFP4 layers continue to use
native block-scaled execution.

## Requirements and warnings

- NVFP4 conversion requires CUDA and a ComfyUI/comfy-kitchen build containing
  `TensorCoreNVFP4Layout`.
- Start from an original BF16/FP16 model, not an already quantized file.
- Quantization is lossy. Keep the source model.
- Test model quality before production use.

## Support development

If this node saves GPU storage or experimentation time, you can support
compatibility testing, documentation, bug fixes, and new architecture support
through [GitHub Sponsors](https://github.com/sponsors/thepororo).

Sponsorship is optional. It never unlocks required node functionality and the
node does not display donation pop-ups or open external pages while running.
See [SUPPORT.md](SUPPORT.md) for the maintainer's story, sponsorship details,
and separate paid implementation support.

## Development note

Designed and maintained by [thepororo](https://github.com/thepororo) with
implementation assistance from OpenAI Codex. AI-assisted changes are reviewed
and validated locally before publication.
