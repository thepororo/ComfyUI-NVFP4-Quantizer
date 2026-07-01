# ComfyUI Universal FP8 / NVFP4 Quantizer

[English](README.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-2b2b2b)](https://www.comfy.org/)
[![후원](https://img.shields.io/badge/후원-GitHub-ea4aaa?logo=githubsponsors)](https://github.com/sponsors/thepororo)

확산 모델 `.safetensors` 파일을 FP8과 NVFP4로 혼합 양자화하는 독립형
ComfyUI 출력 노드입니다.

## 지원 아키텍처

- SDXL
- Wan 2.2
- Qwen Image 및 Qwen Image Edit
- Z-Image
- FLUX 및 FLUX.2
- 범용 Transformer 폴백

## 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thepororo/ComfyUI-NVFP4-Quantizer.git
```

ComfyUI를 재시작하면 `model/quantization`에서
`Universal FP8 / NVFP4 Quantizer`를 찾을 수 있습니다.

## 사용법

1. ComfyUI `models/diffusion_models` 목록에서 모델을 선택합니다.
2. `architecture=auto`, `preset=balanced`, `estimate_only=true`로 설정합니다.
3. `log`에서 감지된 아키텍처와 예상 출력 용량을 확인합니다.
4. `estimate_only`를 끄고 실제 양자화를 실행합니다.

출력은 선택한 원본 모델과 같은 폴더에 저장됩니다. `output_filename`을 비우면
`<원본명>_nvfp4.safetensors`가 사용됩니다.

## 프리셋

- `balanced`: Attention FP8, FFN NVFP4, 기타 대형 Linear FP8
- `quality`: Attention FP8, FFN NVFP4, 기타 Linear BF16
- `aggressive`: Attention과 FFN 모두 NVFP4
- `fp8_all`: 양자화 가능한 대형 Linear를 FP8로 변환
- `custom`: 개별 포맷 설정 사용

## 요구사항과 주의점

- NVFP4 변환에는 CUDA와 `TensorCoreNVFP4Layout`을 제공하는
  ComfyUI/comfy-kitchen 빌드가 필요합니다.
- 이미 양자화된 파일이 아닌 원본 BF16/FP16 모델을 사용하세요.
- 양자화는 손실 변환입니다. 원본을 보관하고 실제 사용 전에 품질을 확인하세요.
- SDXL Conv 가중치는 2D Linear가 아니므로 NVFP4 대상에서 제외됩니다.

## 개발 후원

이 노드가 GPU 저장 공간이나 실험 시간을 줄이는 데 도움이 되었다면
[GitHub Sponsors](https://github.com/sponsors/thepororo)를 통해 호환성 테스트,
문서 작성, 오류 수정, 새로운 아키텍처 지원을 후원할 수 있습니다.

후원은 선택 사항이며 노드의 필수 기능을 잠그지 않습니다. 노드 실행 중 후원
팝업이나 외부 페이지도 표시하지 않습니다. 개발자 소개, 후원 원칙, 별도 유료
구현 지원 안내는 [SUPPORT.md](SUPPORT.md)를 확인해 주세요.

## 개발 참고

[thepororo](https://github.com/thepororo)가 설계하고 관리하며, 구현 과정에서
OpenAI Codex를 활용했습니다. AI 지원 변경 사항은 공개 전에 직접 검토하고 로컬에서
검증했습니다.
