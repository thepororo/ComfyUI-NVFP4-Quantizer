# ComfyUI Universal FP8 / NVFP4 Quantizer

[English](README.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-2b2b2b)](https://www.comfy.org/)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors)](https://github.com/sponsors/thepororo)

확산 모델 `.safetensors`를 FP8과 NVFP4로 혼합 양자화하는 독립형 ComfyUI
출력 노드입니다.

## 지원 아키텍처

SDXL, Wan 2.2, Qwen Image, Qwen Image Edit, Z-Image, FLUX, FLUX.2와
범용 Transformer 폴백을 지원합니다.

## 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thepororo/ComfyUI-NVFP4-Quantizer.git
```

ComfyUI를 재시작하면 `model/quantization`에서 노드를 찾을 수 있습니다.

## 사용법

1. `models/diffusion_models` 목록에서 모델을 선택합니다.
2. `architecture=auto`, `preset=balanced`, `estimate_only=true`로 검사합니다.
3. 로그에서 감지된 아키텍처와 예상 용량을 확인합니다.
4. `estimate_only`를 끄고 실제 변환을 실행합니다.

출력 파일명을 비우면 원본과 같은 폴더에 `<원본명>_nvfp4.safetensors`로 저장됩니다.

## 프리셋

- `balanced`: Attention FP8, FFN NVFP4, 기타 대형 Linear FP8
- `quality`: Attention FP8, FFN NVFP4, 기타 Linear BF16
- `aggressive`: Attention과 FFN NVFP4
- `fp8_all`: 양자화 가능한 대형 Linear FP8
- `custom`: 개별 포맷 설정 사용

## 요구사항과 주의점

- CUDA와 `TensorCoreNVFP4Layout` 지원 ComfyUI/comfy-kitchen이 필요합니다.
- 원본 BF16/FP16 모델을 사용하고 이미 양자화된 파일은 다시 변환하지 마세요.
- 양자화는 손실 변환입니다. 원본을 보관하고 실제 품질을 확인하세요.
- SDXL Conv 가중치는 2D Linear가 아니므로 NVFP4 대상에서 제외됩니다.

## 개발 후원

[GitHub Sponsors](https://github.com/sponsors/thepororo)를 통해 모델 호환성 테스트,
문서화와 유지보수를 후원할 수 있습니다. 후원은 선택 사항이며 필수 기능을
잠금 해제하지 않습니다. 자세한 내용은 [SUPPORT.ko.md](SUPPORT.ko.md)를 확인하세요.

## 개발 참고

[thepororo](https://github.com/thepororo)가 설계하고 관리하며 구현 과정에서
OpenAI Codex를 활용했습니다. AI 지원 변경 사항은 공개 전에 직접 검토하고
로컬에서 검증했습니다.
