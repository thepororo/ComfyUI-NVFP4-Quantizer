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

SDXL, Wan 2.2, Qwen Image/Edit, Z-Image, FLUX, FLUX.2 전용 노드로
분리되었으며 각 노드에서 아키텍처의 주요 레이어 그룹을 개별 설정할 수
있습니다.

- `conservative_nvfp4`(기본값): 민감·입출력 레이어는 원본 dtype을
  유지하고 Attention과 축소 경로는 안전한 FP8 weight-only, 큰
  MLP·확장 경로만 NVFP4로 처리합니다.
- `conservative_fp8`: 민감·입출력 레이어는 원본 dtype을 유지하고
  주요 Linear 가중치를 FP8 weight-only로 저장합니다.
- `custom`: 아키텍처별 모든 레이어 선택 옵션을 사용합니다.

FP8은 암묵적인 activation scale을 사용하지 않고 모델 compute dtype으로
행렬곱합니다. NVFP4 tensor scale은 양수·유한값을 검증하고 네이티브
블록별 FP8 scale과 함께 저장합니다. 이미 양자화된 입력은 거부합니다.

- `balanced`: Attention FP8, FFN NVFP4, 기타 대형 Linear FP8
- `quality`: Attention FP8, FFN NVFP4, 기타 Linear BF16
- `aggressive`: Attention과 FFN NVFP4
- `qwen_edit_14_5gb`: Qwen Image/Edit 2511을 약 14.5GB로 줄이는 프리셋.
  Attention·FFN·기타 대형 Linear는 NVFP4, `img_mod`·`txt_mod`는 FP8,
  입출력 임베딩은 BF16으로 유지
- `qwen_edit_under_14gb`: 약 13.8GB를 목표로 하는 공격적 프리셋.
  처음·마지막 23개 블록의 modulation은 FP8로 보호하고 중간 23–36번
  블록의 modulation만 NVFP4로 변환
- `qwen_edit_perceptual_hybrid`: 용량 제한 없이 지각 품질을 우선하는
  Qwen Image/Edit 2511 프리셋. Attention과 MLP 축소 projection은 FP8
  weight-only, MLP 확장 projection은 NVFP4, modulation은 FP8
  weight-only로 처리하며 입출력, norm, bias는 BF16으로 유지합니다.
  스타일, 표면 질감, 디테일 보존을 우선합니다.
- `fp8_all`: 양자화 가능한 대형 Linear FP8
- `custom`: 개별 포맷 설정 사용

## 요구사항과 주의점

- CUDA와 `TensorCoreNVFP4Layout` 지원 ComfyUI/comfy-kitchen이 필요합니다.
- 원본 BF16/FP16 모델을 사용하고 이미 양자화된 파일은 다시 변환하지 마세요.
- 양자화는 손실 변환입니다. 원본을 보관하고 실제 품질을 확인하세요.
- SDXL Conv 가중치는 2D Linear가 아니므로 NVFP4 대상에서 제외됩니다.

FP8 레이어는 per-tensor weight scale로 저장하고 full-precision matrix
multiplication을 사용합니다. 이 오프라인 변환기는 activation calibration과
`input_scale` 생성을 수행하지 않으므로 activation은 모델의 compute dtype으로
유지합니다. NVFP4 레이어는 네이티브 block-scaled 실행을 계속 사용합니다.

## 개발 후원

[GitHub Sponsors](https://github.com/sponsors/thepororo)를 통해 모델 호환성 테스트,
문서화와 유지보수를 후원할 수 있습니다. 후원은 선택 사항이며 필수 기능을
잠금 해제하지 않습니다. 자세한 내용은 [SUPPORT.ko.md](SUPPORT.ko.md)를 확인하세요.

## 개발 참고

[thepororo](https://github.com/thepororo)가 설계하고 관리하며 구현 과정에서
OpenAI Codex를 활용했습니다. AI 지원 변경 사항은 공개 전에 직접 검토하고
로컬에서 검증했습니다.
