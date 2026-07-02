import json
import subprocess
import sys
from pathlib import Path

import folder_paths

from .quantizer import ARCHITECTURE_PRESETS, FORMATS


COMMON_INPUTS = {
    "model_name": (folder_paths.get_filename_list("diffusion_models"),),
    "output_filename": (
        "STRING",
        {"default": "", "multiline": False, "placeholder": "Empty: source_name_quantized.safetensors"},
    ),
    "preset": (ARCHITECTURE_PRESETS, {"default": "conservative_nvfp4"}),
}


class ArchitectureQuantizer:
    ARCHITECTURE = None
    SUFFIX = "quantized"
    LAYERS = ()
    CONSERVATIVE_NVFP4 = {}
    CONSERVATIVE_FP8 = {}

    @classmethod
    def INPUT_TYPES(cls):
        required = dict(COMMON_INPUTS)
        for name, label, default in cls.LAYERS:
            required[name] = (FORMATS, {"default": default, "tooltip": label})
        required.update({
            "min_elements": (
                "INT", {"default": 65536, "min": 0, "max": 100000000, "step": 1024}
            ),
            "estimate_only": ("BOOLEAN", {"default": False}),
        })
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "log")
    FUNCTION = "quantize"
    CATEGORY = "model/quantization/NVFP"
    OUTPUT_NODE = True

    def quantize(self, model_name, output_filename, preset, min_elements, estimate_only, **kwargs):
        source = Path(folder_paths.get_full_path_or_raise("diffusion_models", model_name)).resolve()
        output_filename = output_filename.strip() or f"{source.stem}_{self.SUFFIX}.safetensors"
        if Path(output_filename).name != output_filename:
            raise ValueError("output_filename must be a file name, not a path")
        if not output_filename.lower().endswith(".safetensors"):
            output_filename += ".safetensors"
        destination = source.with_name(output_filename).resolve()
        if destination.parent != source.parent:
            raise ValueError("Output must be saved beside the selected diffusion model")

        if preset == "conservative_nvfp4":
            layer_config = self.CONSERVATIVE_NVFP4
        elif preset == "conservative_fp8":
            layer_config = self.CONSERVATIVE_FP8
        else:
            layer_config = {name: kwargs[name] for name, _, _ in self.LAYERS}

        script_path = Path(__file__).with_name("quantizer.py")
        command = [
            sys.executable, str(script_path),
            "--input", str(source), "--output", str(destination),
            "--architecture", self.ARCHITECTURE, "--preset", "custom",
            "--profile-name", preset,
            "--attention", "keep", "--ffn", "keep", "--other-linear", "keep",
            "--sensitive", "keep", "--nonquant", "keep",
            "--layer-config", json.dumps(layer_config),
            "--min-elements", str(min_elements),
        ]
        if estimate_only:
            command.append("--estimate-only")
        result = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        log = "\n".join(x for x in (result.stdout, result.stderr) if x).strip()
        if result.returncode:
            raise RuntimeError(f"Quantizer failed ({result.returncode})\n{log}")
        return str(destination), log


class NVFP4SDXLQuantizer(ArchitectureQuantizer):
    ARCHITECTURE = "sdxl"
    SUFFIX = "sdxl_nvfp"
    LAYERS = (
        ("attention", "Attention projections", "fp8"),
        ("ffn", "Feed-forward Linear layers", "nvfp4"),
        ("other_linear", "Other large Linear layers", "keep"),
        ("sensitive", "Embeddings and input/output layers", "keep"),
    )
    CONSERVATIVE_NVFP4 = {"attention": "fp8", "ffn": "nvfp4", "other_linear": "keep", "sensitive": "keep"}
    CONSERVATIVE_FP8 = {"attention": "fp8", "ffn": "fp8", "other_linear": "keep", "sensitive": "keep"}


class NVFP4Wan22Quantizer(ArchitectureQuantizer):
    ARCHITECTURE = "wan22"
    SUFFIX = "wan22_nvfp"
    LAYERS = (
        ("self_attention", "Self-attention projections", "fp8"),
        ("cross_attention", "Cross-attention projections", "fp8"),
        ("ffn", "Feed-forward Linear layers", "nvfp4"),
        ("other_linear", "Other large Linear layers", "keep"),
        ("sensitive", "Embeddings, modulation, and input/output layers", "keep"),
    )
    CONSERVATIVE_NVFP4 = {"self_attention": "fp8", "cross_attention": "fp8", "ffn": "nvfp4", "other_linear": "keep", "sensitive": "keep"}
    CONSERVATIVE_FP8 = {"self_attention": "fp8", "cross_attention": "fp8", "ffn": "fp8", "other_linear": "keep", "sensitive": "keep"}


class NVFP4QwenImageQuantizer(ArchitectureQuantizer):
    ARCHITECTURE = "qwen_image_edit"
    SUFFIX = "qwen_nvfp"
    LAYERS = (
        ("attention", "Image/text Attention projections", "fp8"),
        ("mlp_expansion", "Image/text MLP expansion projections", "nvfp4"),
        ("mlp_reduction", "Image/text MLP reduction projections", "fp8"),
        ("modulation", "Image/text modulation layers", "fp8"),
        ("other_linear", "Other large Linear layers", "keep"),
        ("sensitive", "Input/output and sensitive layers", "keep"),
    )
    CONSERVATIVE_NVFP4 = {"attention": "fp8", "mlp_expansion": "nvfp4", "mlp_reduction": "fp8", "modulation": "fp8", "other_linear": "keep", "sensitive": "keep"}
    CONSERVATIVE_FP8 = {"attention": "fp8", "mlp_expansion": "fp8", "mlp_reduction": "fp8", "modulation": "fp8", "other_linear": "keep", "sensitive": "keep"}


class NVFP4ZImageQuantizer(ArchitectureQuantizer):
    ARCHITECTURE = "z_image"
    SUFFIX = "zimage_nvfp"
    LAYERS = (
        ("attention", "Attention QKV/output projections", "fp8"),
        ("ffn", "Feed-forward W1/W2/W3 layers", "nvfp4"),
        ("other_linear", "Other large Linear layers", "keep"),
        ("sensitive", "Embeddings and input/output layers", "keep"),
    )
    CONSERVATIVE_NVFP4 = {"attention": "fp8", "ffn": "nvfp4", "other_linear": "keep", "sensitive": "keep"}
    CONSERVATIVE_FP8 = {"attention": "fp8", "ffn": "fp8", "other_linear": "keep", "sensitive": "keep"}


class NVFP4FluxQuantizer(ArchitectureQuantizer):
    ARCHITECTURE = "flux"
    SUFFIX = "flux_nvfp"
    LAYERS = (
        ("dual_attention", "Double-stream image/text Attention", "fp8"),
        ("dual_mlp", "Double-stream image/text MLP", "nvfp4"),
        ("single_expansion", "Single-stream expansion/combined projection", "nvfp4"),
        ("single_reduction", "Single-stream output projection", "fp8"),
        ("other_linear", "Other large Linear layers", "keep"),
        ("sensitive", "Modulation and input/output layers", "keep"),
    )
    CONSERVATIVE_NVFP4 = {"dual_attention": "fp8", "dual_mlp": "nvfp4", "single_expansion": "nvfp4", "single_reduction": "fp8", "other_linear": "keep", "sensitive": "keep"}
    CONSERVATIVE_FP8 = {"dual_attention": "fp8", "dual_mlp": "fp8", "single_expansion": "fp8", "single_reduction": "fp8", "other_linear": "keep", "sensitive": "keep"}


class NVFP4Flux2Quantizer(NVFP4FluxQuantizer):
    ARCHITECTURE = "flux2"
    SUFFIX = "flux2_nvfp"


NODE_CLASS_MAPPINGS = {
    "NVFP4SDXLQuantizer": NVFP4SDXLQuantizer,
    "NVFP4Wan22Quantizer": NVFP4Wan22Quantizer,
    "NVFP4QwenImageQuantizer": NVFP4QwenImageQuantizer,
    "NVFP4ZImageQuantizer": NVFP4ZImageQuantizer,
    "NVFP4FluxQuantizer": NVFP4FluxQuantizer,
    "NVFP4Flux2Quantizer": NVFP4Flux2Quantizer,
    "NVFP4UniversalQuantizer": NVFP4QwenImageQuantizer,
    "NVFP4WanQuantizer": NVFP4Wan22Quantizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NVFP4SDXLQuantizer": "NVFP Quantizer — SDXL",
    "NVFP4Wan22Quantizer": "NVFP Quantizer — Wan 2.2",
    "NVFP4QwenImageQuantizer": "NVFP Quantizer — Qwen Image / Edit",
    "NVFP4ZImageQuantizer": "NVFP Quantizer — Z-Image",
    "NVFP4FluxQuantizer": "NVFP Quantizer — FLUX",
    "NVFP4Flux2Quantizer": "NVFP Quantizer — FLUX.2",
    "NVFP4UniversalQuantizer": "NVFP Quantizer — Qwen Image / Edit (Legacy ID)",
    "NVFP4WanQuantizer": "NVFP Quantizer — Wan 2.2 (Legacy ID)",
}
