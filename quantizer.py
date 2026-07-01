import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ARCHITECTURES = (
    "auto",
    "sdxl",
    "wan22",
    "qwen_image",
    "qwen_image_edit",
    "z_image",
    "flux",
    "flux2",
    "generic_transformer",
)
FORMATS = ("keep", "bf16", "fp16", "fp32", "fp8", "nvfp4")
REGULAR_FORMATS = ("keep", "bf16", "fp16", "fp32")
PRESETS = ("balanced", "quality", "aggressive", "fp8_all", "custom")

PRESET_FORMATS = {
    "balanced": ("fp8", "nvfp4", "fp8", "bf16", "bf16"),
    "quality": ("fp8", "nvfp4", "bf16", "bf16", "bf16"),
    "aggressive": ("nvfp4", "nvfp4", "fp8", "bf16", "bf16"),
    "fp8_all": ("fp8", "fp8", "fp8", "bf16", "bf16"),
}

PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def ensure_comfy_import_path():
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "comfy" / "quant_ops.py").is_file():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return candidate
    raise RuntimeError(
        "Could not locate the ComfyUI root containing comfy/quant_ops.py. "
        "Install this folder under ComfyUI/custom_nodes."
    )


def normalize_key(key):
    for prefix in PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def detect_architecture(keys):
    names = [normalize_key(k) for k in keys]
    checks = (
        ("flux2", lambda k: k.startswith("double_stream_modulation_img.lin.")),
        ("flux", lambda k: "double_blocks." in k or "single_blocks." in k),
        ("wan22", lambda k: re.search(r"(?:^|\.)blocks\.\d+\.self_attn\.", k)),
        ("qwen_image", lambda k: "transformer_blocks." in k and ".attn.to_q." in k),
        ("z_image", lambda k: re.search(r"(?:^|\.)layers\.\d+\.attention\.(?:qkv|out)\.", k)),
        ("sdxl", lambda k: k.startswith(("input_blocks.", "middle_block.", "output_blocks."))),
    )
    for architecture, predicate in checks:
        if any(predicate(k) for k in names):
            return architecture
    return "generic_transformer"


def is_sensitive(key):
    k = normalize_key(key).lower()
    tokens = (
        "patch_embedding", "patch_embed", "img_in", "txt_in", "time_in",
        "vector_in", "guidance_in", "time_text_embed", "time_embed",
        "t_embedder", "x_embedder", "context_embedder", "caption_projection",
        "proj_out", "final_layer", "final_linear", "out_layer",
        "ada_ln", "adaln", "modulation", "img_mod", "txt_mod",
    )
    return any(token in k for token in tokens)


def classify_weight(key, tensor, architecture):
    if not key.endswith(".weight") or not tensor.is_floating_point() or tensor.ndim != 2:
        return "nonquant"
    if is_sensitive(key):
        return "sensitive"

    k = normalize_key(key)
    if architecture == "wan22":
        if re.search(r"blocks\.\d+\.(?:self_attn|cross_attn)\.(?:q|k|v|o)\.weight$", k):
            return "attention"
        if re.search(r"blocks\.\d+\.ffn\.(?:0|2)\.weight$", k):
            return "ffn"
    elif architecture in ("flux", "flux2"):
        if re.search(r"double_blocks\.\d+\.(?:img_attn|txt_attn)\.(?:qkv|proj)\.weight$", k):
            return "attention"
        if re.search(r"double_blocks\.\d+\.(?:img_mlp|txt_mlp)\.(?:0|2)\.weight$", k):
            return "ffn"
        if re.search(r"single_blocks\.\d+\.linear1\.weight$", k):
            return "attention"
        if re.search(r"single_blocks\.\d+\.linear2\.weight$", k):
            return "ffn"
    elif architecture in ("qwen_image", "qwen_image_edit"):
        if re.search(r"transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|to_out\.0|to_add_out)\.weight$", k):
            return "attention"
        if re.search(r"transformer_blocks\.\d+\.(?:img_mlp|txt_mlp)\.", k):
            return "ffn"
    elif architecture == "z_image":
        if re.search(r"layers\.\d+\.attention\.(?:qkv|out)\.weight$", k):
            return "attention"
        if re.search(r"layers\.\d+\.feed_forward\.(?:w1|w2|w3)\.weight$", k):
            return "ffn"
    elif architecture == "sdxl":
        if re.search(r"(?:attn1|attn2)\.(?:to_q|to_k|to_v|to_out\.0)\.weight$", k):
            return "attention"
        if re.search(r"\.ff\.net\.(?:0\.proj|2)\.weight$", k):
            return "ffn"

    lower = k.lower()
    if re.search(r"(?:^|\.)(?:attn|attention|self_attn|cross_attn)\.", lower) and re.search(
        r"(?:qkv|to_q|to_k|to_v|to_out|query|key|value|proj|out)\.weight$", lower
    ):
        return "attention"
    if re.search(r"(?:^|\.)(?:ffn|feed_forward|mlp|ff)\.", lower):
        return "ffn"
    return "other_linear"


def tensor_bytes(tensor, fmt):
    if fmt == "keep":
        return tensor.numel() * tensor.element_size()
    if fmt in ("bf16", "fp16"):
        return tensor.numel() * 2
    if fmt == "fp32":
        return tensor.numel() * 4
    if fmt == "fp8":
        return tensor.numel() + 4
    if fmt == "nvfp4":
        rows, cols = tensor.shape
        return rows * math.ceil(cols / 2) + rows * math.ceil(cols / 16) + 4
    raise ValueError(fmt)


def cast_regular(tensor, fmt):
    if fmt == "keep" or not tensor.is_floating_point():
        return tensor.cpu().contiguous()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[fmt]
    return tensor.to(dtype).cpu().contiguous()


def quantize_fp8(tensor):
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 quantization requires CUDA")
    ensure_comfy_import_path()
    from comfy.quant_ops import TensorCoreFP8E4M3Layout
    x = tensor.to("cuda", non_blocking=True).contiguous()
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        x = x.to(torch.bfloat16)
    with torch.no_grad():
        qdata, params = TensorCoreFP8E4M3Layout.quantize(
            x, scale="recalculate", stochastic_rounding=0, inplace_ops=False
        )
    return (
        qdata.detach().cpu().contiguous(),
        params.scale.detach().cpu().to(torch.float32).reshape(()).contiguous(),
    )


def quantize_nvfp4(tensor):
    if tensor.ndim != 2:
        raise ValueError("NVFP4 only supports 2D Linear weights")
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 quantization requires CUDA")
    ensure_comfy_import_path()
    from comfy.quant_ops import TensorCoreNVFP4Layout
    x = tensor.to("cuda", non_blocking=True).contiguous()
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        x = x.to(torch.bfloat16)
    with torch.no_grad():
        qdata, params = TensorCoreNVFP4Layout.quantize(
            x, scale="recalculate", stochastic_rounding=0, inplace_ops=False
        )
    return (
        qdata.detach().cpu().contiguous(),
        params.block_scale.detach().cpu().contiguous(),
        params.scale.detach().cpu().to(torch.float32).reshape(()).contiguous(),
    )


def resolved_formats(preset, attention, ffn, other_linear, sensitive, nonquant):
    if preset == "custom":
        return {
            "attention": attention, "ffn": ffn, "other_linear": other_linear,
            "sensitive": sensitive, "nonquant": nonquant,
        }
    values = PRESET_FORMATS[preset]
    return dict(zip(("attention", "ffn", "other_linear", "sensitive", "nonquant"), values))


def quantize_file(
    input_path, output_path, architecture="auto", preset="balanced",
    attention="fp8", ffn="nvfp4", other_linear="fp8",
    sensitive="bf16", nonquant="bf16", min_elements=65536,
    estimate_only=False,
):
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise ValueError(f"Input must be an existing .safetensors file: {source}")
    if destination.suffix.lower() != ".safetensors":
        raise ValueError("Output path must end with .safetensors")
    if source == destination and not estimate_only:
        raise ValueError("Input and output paths must be different")

    formats = resolved_formats(preset, attention, ffn, other_linear, sensitive, nonquant)
    out_sd, quant_layers = {}, {}
    format_count, pair_count = Counter(), Counter()
    format_bytes = defaultdict(int)

    with safe_open(str(source), framework="pt", device="cpu") as handle:
        old_metadata = handle.metadata() or {}
        keys = list(handle.keys())
        detected = detect_architecture(keys)
        selected = detected if architecture == "auto" else architecture
        classification_arch = "qwen_image" if selected == "qwen_image_edit" else selected
        print(f"architecture: {selected} (detected: {detected})")
        print("formats:", ", ".join(f"{k}={v}" for k, v in formats.items()))

        input_bytes = output_bytes = 0
        for index, key in enumerate(keys, 1):
            tensor = handle.get_tensor(key)
            input_bytes += tensor.numel() * tensor.element_size()
            category = classify_weight(key, tensor, classification_arch)
            fmt = formats[category]
            if category not in ("sensitive", "nonquant") and tensor.numel() < min_elements:
                category, fmt = "sensitive", formats["sensitive"]
            if fmt in ("fp8", "nvfp4") and (tensor.ndim != 2 or not tensor.is_floating_point()):
                fmt = formats["nonquant"]

            format_count[fmt] += 1
            pair_count[(category, fmt)] += 1
            estimated = tensor_bytes(tensor, fmt)
            format_bytes[fmt] += estimated
            output_bytes += estimated

            if estimate_only:
                continue
            if fmt == "fp8":
                q, scale = quantize_fp8(tensor)
                out_sd[key], out_sd[key + "_scale"] = q, scale
                quant_layers[normalize_key(key)[:-7]] = {"format": "float8_e4m3fn"}
            elif fmt == "nvfp4":
                q, scale, scale2 = quantize_nvfp4(tensor)
                out_sd[key], out_sd[key + "_scale"], out_sd[key + "_scale_2"] = q, scale, scale2
                quant_layers[normalize_key(key)[:-7]] = {"format": "nvfp4"}
            else:
                out_sd[key] = cast_regular(tensor, fmt)
            if index % 50 == 0:
                print(f"processed {index}/{len(keys)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n=== summary ===")
    print(f"input estimate : {input_bytes / 1024**3:.3f} GiB")
    print(f"output estimate: {output_bytes / 1024**3:.3f} GiB")
    print(f"reduction      : {100 * (1 - output_bytes / input_bytes):.2f}%")
    for (category, fmt), count in sorted(pair_count.items()):
        print(f"  {category:12} {fmt:6}: {count}")

    if estimate_only:
        print("No file saved (estimate_only).")
        return str(destination)

    metadata = dict(old_metadata)
    metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": quant_layers}, ensure_ascii=False
    )
    metadata.update({
        "converted_by": "ComfyUI-NVFP4-Universal-Quantizer",
        "architecture": selected,
        "detected_architecture": detected,
        "preset": preset,
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(destination), metadata=metadata)
    print(f"saved: {destination} ({destination.stat().st_size / 1024**3:.3f} GiB)")
    return str(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="auto")
    parser.add_argument("--preset", choices=PRESETS, default="balanced")
    parser.add_argument("--attention", choices=FORMATS, default="fp8")
    parser.add_argument("--ffn", choices=FORMATS, default="nvfp4")
    parser.add_argument("--other-linear", choices=FORMATS, default="fp8")
    parser.add_argument("--sensitive", choices=REGULAR_FORMATS, default="bf16")
    parser.add_argument("--nonquant", choices=REGULAR_FORMATS, default="bf16")
    parser.add_argument("--min-elements", type=int, default=65536)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    quantize_file(
        args.input, args.output, args.architecture, args.preset,
        args.attention, args.ffn, args.other_linear, args.sensitive,
        args.nonquant, args.min_elements, args.estimate_only,
    )


if __name__ == "__main__":
    main()
