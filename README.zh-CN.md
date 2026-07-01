# ComfyUI Universal FP8 / NVFP4 Quantizer

[English](README.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

用于扩散模型 `.safetensors` 文件的独立 ComfyUI FP8/NVFP4 混合量化输出节点。

## 支持的架构

- SDXL
- Wan 2.2
- Qwen Image 与 Qwen Image Edit
- Z-Image
- FLUX 与 FLUX.2
- 通用 Transformer 回退分类

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thepororo/ComfyUI-NVFP4-Quantizer.git
```

重启 ComfyUI。节点位于 `model/quantization`，名称为
`Universal FP8 / NVFP4 Quantizer`。

## 使用方法

1. 从 ComfyUI 的 `models/diffusion_models` 列表选择模型。
2. 设置 `architecture=auto`、`preset=balanced`、`estimate_only=true`。
3. 在 `log` 中确认检测到的架构和预计输出大小。
4. 关闭 `estimate_only` 后执行实际量化。

输出文件会保存在源模型旁边。`output_filename` 留空时自动使用
`<源文件名>_nvfp4.safetensors`。

## 预设

- `balanced`：注意力 FP8、FFN NVFP4、其他大型 Linear 层 FP8。
- `quality`：注意力 FP8、FFN NVFP4、其他 Linear 层 BF16。
- `aggressive`：注意力和 FFN 均使用 NVFP4。
- `fp8_all`：符合条件的大型 Linear 层使用 FP8。
- `custom`：使用各个格式选项。

## 要求与注意事项

- NVFP4 转换需要 CUDA，以及包含 `TensorCoreNVFP4Layout` 的
  ComfyUI/comfy-kitchen。
- 请使用原始 BF16/FP16 模型，不要重复量化。
- 量化有损，请保留源模型并在正式使用前测试质量。
- SDXL 卷积权重不是 2D Linear，因此不会转换为 NVFP4。

## 开发说明

本项目由 [thepororo](https://github.com/thepororo) 设计和维护，并使用
OpenAI Codex 协助实现。所有 AI 辅助修改均在发布前经过人工审查和本地验证。
