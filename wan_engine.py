"""Wan 2.2 TI2V-5B @ 720p — best open model for RTX 4090 (Kling-class self-host)."""
from __future__ import annotations

import bootstrap  # noqa: F401 — patch diffusers before import

import os
import shutil
import subprocess
from pathlib import Path

_pipe = None
_model_id: str | None = None
_lora_loaded: bool = False

# Default: Wan 2.2 5B — 720p@24fps on consumer 4090 (Apache 2.0).
WAN_MODEL_ID = os.environ.get(
    "WAN_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
)
WAN_MODEL_LITE = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_MODEL_PRO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

# 720p 16:9 — official Wan 2.2 TI2V resolution.
W22_WIDTH = 1280
W22_HEIGHT = 704
W22_FPS = 24
W22_FRAMES = 97  # ~4s @ 24fps (4n+1)
W22_STEPS = 40
W22_GUIDANCE = 5.0

# Fallback 2.1 1.3B 480p.
LITE_WIDTH = 832
LITE_HEIGHT = 480
LITE_FPS = 16
LITE_FRAMES = 41
LITE_STEPS = 32

NEGATIVE_PROMPT = (
    "oversaturated, overexposed, static, blurry details, subtitles, artwork, painting, "
    "still image, grayish, worst quality, low quality, JPEG artifacts, ugly, "
    "distorted, flickering, jitter, deformed, cartoon, illustration, artificial, "
    "pixelated, out of focus, watermark, text"
)

REALISM_SUFFIX = (
    "photorealistic, cinematic lighting, natural smooth motion, sharp focus, "
    "realistic textures, lifelike, high detail, professional color grading"
)

# Optional LoRA trained via training/ads-lora/ (mount on RunPod network volume).
WAN_ADS_LORA_PATH = os.environ.get("WAN_ADS_LORA_PATH", "").strip()
WAN_ADS_LORA_SCALE = float(os.environ.get("WAN_ADS_LORA_SCALE", "0.85"))


def enhance_prompt(prompt: str) -> str:
    p = prompt.strip()
    if not p:
        return p
    lower = p.lower()
    if any(
        k in lower
        for k in ("photorealistic", "cinematic", "4k", "lifelike", "sharp focus")
    ):
        return p
    return f"{p}, {REALISM_SUFFIX}"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _is_wan22(model_id: str) -> bool:
    return "2.2" in model_id or "Wan2.2" in model_id


def _is_pro_model(model_id: str) -> bool:
    return "A14B" in model_id or "14B" in model_id


def _cuda_ready() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _clear_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _inference_profile(model_id: str) -> tuple[int, int, int, int, int]:
    """frames, width, height, steps, fps."""
    if _is_wan22(model_id) and not _is_pro_model(model_id):
        _log(
            f"[wan_engine] Wan2.2 TI2V-5B 720p: {W22_FRAMES}f {W22_STEPS}steps "
            f"@ {W22_WIDTH}x{W22_HEIGHT}"
        )
        return W22_FRAMES, W22_WIDTH, W22_HEIGHT, W22_STEPS, W22_FPS

    if _is_pro_model(model_id):
        vram_gb = _gpu_vram_gb()
        if vram_gb >= 48:
            return 49, 832, 480, 30, 16
        return 33, 832, 480, 28, 16

    return LITE_FRAMES, LITE_WIDTH, LITE_HEIGHT, LITE_STEPS, LITE_FPS


def generate_smoke_video(prompt: str, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    safe = prompt.replace(":", r"\:").replace("'", r"\'")[:120]
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x1a1a2e:s={W22_WIDTH}x{W22_HEIGHT}:d=2",
            "-vf",
            f"drawtext=text='SMOKE {safe}':fontcolor=white:fontsize=18:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
        timeout=60,
    )


def _check_model_disk(model_id: str) -> None:
    cache = os.environ.get("HF_HOME", "/tmp/huggingface")
    os.makedirs(cache, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / (1024**3)
    if _is_wan22(model_id):
        need_gb = 25.0
    elif _is_pro_model(model_id):
        need_gb = 45.0
    else:
        need_gb = 12.0
    _log(f"[wan_engine] HF_HOME={cache} free={free_gb:.1f}GB need={need_gb:.0f}GB")
    if free_gb < need_gb:
        raise RuntimeError(
            f"Not enough disk at {cache}: {free_gb:.1f}GB free, need ~{need_gb:.0f}GB. "
            "Set RunPod Container Disk to 80GB."
        )


def _configure_pipe(pipe, model_id: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU not available on worker")

    vram_gb = _gpu_vram_gb()
    _log(f"[wan_engine] GPU {torch.cuda.get_device_name(0)} VRAM={vram_gb:.1f}GB")
    _clear_cuda()

    if _is_wan22(model_id) and not _is_pro_model(model_id):
        # 5B: fits 4090 with light offload if needed.
        try:
            pipe.enable_attention_slicing("auto")
        except Exception:
            pass
        if vram_gb >= 20:
            _log("[wan_engine] Wan2.2 5B → pipe.to(cuda) [4090 720p]")
            pipe.to("cuda")
        else:
            _log("[wan_engine] Wan2.2 5B → model_cpu_offload")
            pipe.enable_model_cpu_offload()
    elif _is_pro_model(model_id):
        try:
            pipe.enable_attention_slicing("max")
            pipe.enable_vae_slicing()
        except Exception:
            pass
        if vram_gb >= 48:
            pipe.to("cuda")
        else:
            pipe.enable_sequential_cpu_offload()
    else:
        try:
            pipe.enable_attention_slicing("auto")
        except Exception:
            pass
        pipe.to("cuda") if vram_gb >= 10 else pipe.enable_model_cpu_offload()

    _clear_cuda()
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        _log(f"[wan_engine] CUDA allocated: {alloc_gb:.2f}GB")


def _maybe_load_ads_lora(pipe) -> None:
    global _lora_loaded
    if _lora_loaded or not WAN_ADS_LORA_PATH:
        return
    lora_path = Path(WAN_ADS_LORA_PATH)
    if not lora_path.exists():
        _log(f"[wan_engine] WAN_ADS_LORA_PATH missing: {lora_path}")
        return
    try:
        if lora_path.is_file():
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        else:
            pipe.load_lora_weights(str(lora_path))
        _lora_loaded = True
        _log(f"[wan_engine] Ads LoRA loaded from {lora_path} scale={WAN_ADS_LORA_SCALE}")
    except Exception as e:
        _log(f"[wan_engine] Ads LoRA load failed: {e}")


def _download_product_image(url: str):
    import requests
    from PIL import Image
    from io import BytesIO

    _log(f"[wan_engine] downloading product image…")
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    _log(f"[wan_engine] product image {img.size[0]}x{img.size[1]}")
    return img


def _load_pipeline(model_id: str):
    global _pipe, _model_id
    if _pipe is not None and _model_id == model_id:
        _log("[wan_engine] pipeline cache hit")
        return _pipe

    _check_model_disk(model_id)
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    _log(f"[wan_engine] loading {model_id}...")
    _clear_cuda()

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    _configure_pipe(pipe, model_id)
    _maybe_load_ads_lora(pipe)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    _pipe = pipe
    _model_id = model_id
    _log(f"[wan_engine] ready ({model_id})")
    return _pipe


def warmup(model_id: str | None = None) -> None:
    if not _cuda_ready():
        _log("[wan_engine] warmup skipped — no CUDA")
        return
    mid = model_id or WAN_MODEL_ID
    _load_pipeline(mid)
    _log("[wan_engine] warmup done")


def generate_video(
    prompt: str,
    duration_sec: int,
    output_path: Path,
    model_id: str = WAN_MODEL_ID,
    *,
    generation_mode: str = "standard",
    ad_category: str | None = None,
    product_image_url: str | None = None,
) -> None:
    if not _cuda_ready():
        raise RuntimeError("CUDA/GPU not available on worker")

    from diffusers.utils import export_to_video

    ads_mode = str(generation_mode or "standard").strip().lower() == "ads"
    mode_label = "ads" if ads_mode else "standard"
    _log(
        f"[wan_engine] generate mode={mode_label} model={model_id} "
        f"prompt={prompt[:60]!r}"
    )
    _clear_cuda()

    pipe = _load_pipeline(model_id)

    if ads_mode:
        from ads_prompts import ads_negative_prompt, build_ads_prompt

        enhanced = build_ads_prompt(prompt, ad_category)
        negative = ads_negative_prompt(NEGATIVE_PROMPT)
    else:
        enhanced = enhance_prompt(prompt)
        negative = NEGATIVE_PROMPT

    num_frames, width, height, steps, fps = _inference_profile(model_id)

    # Cap frames to requested duration when shorter than profile max.
    if duration_sec > 0 and fps > 0:
        want = int(duration_sec * fps)
        want = max(9, min(want, num_frames))
        want = (want // 4) * 4 + 1
        num_frames = want

    product_image = None
    if product_image_url:
        try:
            product_image = _download_product_image(product_image_url)
        except Exception as e:
            _log(f"[wan_engine] product image failed ({e}) — text-only fallback")

    pipe_kwargs: dict = {
        "prompt": enhanced,
        "negative_prompt": negative,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": W22_GUIDANCE if _is_wan22(model_id) else 6.0,
    }
    if product_image is not None:
        pipe_kwargs["image"] = product_image
        _log("[wan_engine] TI2V — product image conditioning")

    if ads_mode and _lora_loaded and WAN_ADS_LORA_SCALE != 1.0:
        try:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": WAN_ADS_LORA_SCALE}
        except Exception:
            pass

    _log(f"[wan_engine] infer {num_frames}f {width}x{height} {steps}steps")
    result = pipe(**pipe_kwargs)

    export_to_video(result.frames[0], str(output_path), fps=fps)
    _log(f"[wan_engine] done → {output_path} (~{num_frames / fps:.1f}s @ {fps}fps)")
