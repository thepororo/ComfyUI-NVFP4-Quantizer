import subprocess
import sys
from pathlib import Path

import folder_paths

from .quantizer import ARCHITECTURES, FORMATS, PRESETS, REGULAR_FORMATS


class NVFP4UniversalQuantizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "output_filename": (
                    "STRING",
                    {"default": "", "multiline": False, "placeholder": "비워두면 원본명_nvfp4.safetensors"},
                ),
                "architecture": (ARCHITECTURES, {"default": "auto"}),
                "preset": (PRESETS, {"default": "balanced"}),
                "attention": (FORMATS, {"default": "fp8"}),
                "ffn": (FORMATS, {"default": "nvfp4"}),
                "other_linear": (FORMATS, {"default": "fp8"}),
                "sensitive": (REGULAR_FORMATS, {"default": "bf16"}),
                "nonquant": (REGULAR_FORMATS, {"default": "bf16"}),
                "min_elements": (
                    "INT", {"default": 65536, "min": 0, "max": 100000000, "step": 1024}
                ),
                "estimate_only": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "log")
    FUNCTION = "quantize"
    CATEGORY = "model/quantization"
    OUTPUT_NODE = True

    def quantize(
        self, model_name, output_filename, architecture, preset, attention, ffn,
        other_linear, sensitive, nonquant, min_elements, estimate_only,
    ):
        script_path = Path(__file__).with_name("quantizer.py")
        source = Path(
            folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        ).resolve()

        output_filename = output_filename.strip()
        if not output_filename:
            output_filename = f"{source.stem}_nvfp4.safetensors"
        if Path(output_filename).name != output_filename:
            raise ValueError("output_filename에는 폴더가 아닌 파일명만 입력해야 합니다.")
        if not output_filename.lower().endswith(".safetensors"):
            output_filename += ".safetensors"
        destination = source.with_name(output_filename).resolve()
        if destination.parent != source.parent:
            raise ValueError("출력 파일은 선택한 확산 모델과 같은 폴더에 저장되어야 합니다.")

        command = [
            sys.executable, str(script_path),
            "--input", str(source), "--output", str(destination),
            "--architecture", architecture, "--preset", preset,
            "--attention", attention, "--ffn", ffn,
            "--other-linear", other_linear, "--sensitive", sensitive,
            "--nonquant", nonquant, "--min-elements", str(min_elements),
        ]
        if estimate_only:
            command.append("--estimate-only")
        result = subprocess.run(
            command, cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        log = "\n".join(x for x in (result.stdout, result.stderr) if x).strip()
        if result.returncode:
            raise RuntimeError(f"Quantizer failed ({result.returncode})\n{log}")
        return str(destination), log


# Keep the legacy type ID discoverable; existing workflows should review inputs.
NODE_CLASS_MAPPINGS = {
    "NVFP4UniversalQuantizer": NVFP4UniversalQuantizer,
    "NVFP4WanQuantizer": NVFP4UniversalQuantizer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NVFP4UniversalQuantizer": "Universal FP8 / NVFP4 Quantizer",
    "NVFP4WanQuantizer": "Universal FP8 / NVFP4 Quantizer (Legacy ID)",
}
