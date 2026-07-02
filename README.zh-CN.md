# ComfyUI Universal FP8 / NVFP4 Quantizer

[English](README.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-2b2b2b)](https://www.comfy.org/)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors)](https://github.com/sponsors/thepororo)

用于将扩散模型 `.safetensors` 文件混合量化为 FP8/NVFP4 的独立 ComfyUI
输出节点。

## 支持的架构

支持 SDXL、Wan 2.2、Qwen Image、Qwen Image Edit、Z-Image、FLUX、
FLUX.2 和通用 Transformer 回退分类。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thepororo/ComfyUI-NVFP4-Quantizer.git
```

重启 ComfyUI 后，可在 `model/quantization` 中找到该节点。

## 使用方法

1. 从 `models/diffusion_models` 列表选择模型。
2. 使用 `architecture=auto`、`preset=balanced`、`estimate_only=true` 检查。
3. 在日志中确认架构和预计输出大小。
4. 关闭 `estimate_only` 后执行实际转换。

输出文件名留空时，会在源模型旁保存为 `<源文件名>_nvfp4.safetensors`。

## 预设

节点已按 SDXL、Wan 2.2、Qwen Image/Edit、Z-Image、FLUX 和 FLUX.2
拆分，每个节点都可以单独设置该架构的主要层组。

- `conservative_nvfp4`（默认）：敏感层和输入输出层保持源 dtype，
  Attention 与缩减路径使用安全的 FP8 weight-only，仅大型 MLP/扩展
  路径使用 NVFP4。
- `conservative_fp8`：敏感层和输入输出层保持源 dtype，符合条件的主要
  Linear 权重使用 FP8 weight-only。
- `custom`：启用所有架构专用的层选择选项。

FP8 不使用隐式 activation scale，矩阵乘法保持模型 compute dtype。
NVFP4 tensor scale 会强制为正数、检查有限值，并与原生逐块 FP8 scale
一起保存。已量化的输入会被拒绝。

- `balanced`：注意力 FP8、FFN NVFP4、其他大型 Linear 层 FP8
- `quality`：注意力 FP8、FFN NVFP4、其他 Linear 层 BF16
- `aggressive`：注意力和 FFN 使用 NVFP4
- `qwen_edit_14_5gb`：针对 Qwen Image/Edit 2511 的约 14.5GB 预设。
  注意力、FFN 和其他大型 Linear 层使用 NVFP4，`img_mod`、`txt_mod`
  使用 FP8，输入输出嵌入保持 BF16
- `qwen_edit_under_14gb`：目标约 13.8GB 的激进预设。前后各 23 个块的
  modulation 保持 FP8，仅将中间第 23–36 块的 modulation 转换为 NVFP4
- `qwen_edit_perceptual_hybrid`：不设文件大小目标、优先感知质量的
  Qwen Image/Edit 2511 预设。Attention 和 MLP 缩减投影使用 FP8
  weight-only，MLP 扩展投影使用 NVFP4，modulation 使用 FP8
  weight-only，输入输出层、norm 和 bias 保持 BF16，优先保留风格、
  表面质感和细节。
- `fp8_all`：符合条件的大型 Linear 层使用 FP8
- `custom`：使用各个格式选项

## 要求与注意事项

- 需要 CUDA，以及支持 `TensorCoreNVFP4Layout` 的 ComfyUI/comfy-kitchen。
- 使用原始 BF16/FP16 模型，不要重复量化。
- 量化是有损转换。请保留源模型并测试实际质量。
- SDXL 卷积权重不是 2D Linear，因此不会转换为 NVFP4。

FP8 层使用 per-tensor weight scale 存储，并启用 full-precision matrix
multiplication。该离线转换器不执行 activation calibration，也不生成
`input_scale`，因此 activation 保持模型的 compute dtype。NVFP4 层继续使用
原生 block-scaled 执行。

## 支持开发

可以通过 [GitHub Sponsors](https://github.com/sponsors/thepororo) 支持模型兼容性测试、
文档和维护。赞助完全自愿，不会解锁必要功能。详情请参阅
[SUPPORT.zh-CN.md](SUPPORT.zh-CN.md)。

## 开发说明

本项目由 [thepororo](https://github.com/thepororo) 设计和维护，并使用
OpenAI Codex 协助实现。所有 AI 辅助修改均在发布前经过人工审查和本地验证。
