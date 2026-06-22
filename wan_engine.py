"""Wan T2V — singleton pipeline kept hot between jobs."""
from __future__ import annotations

import bootstrap  # noqa: F401 — patch diffusers before import

import os
import shutil
import subprocess
from pathlib import Path

_pipe = None
_model_id: str | None = None

# Pro quality — 14B (much sharper than 1.3B). Override: WAN_MODEL_ID env.
WAN_MODEL_ID = os.environ.get(
    "WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers"
)
WAN_MODEL_LITE = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# 480p cinematic — Wan uses num_frames = 4n+1
PRO_FRAMES = 49
PRO_STEPS = 30
PRO_WIDTH = 832
PRO_HEIGHT = 480
PRO_FPS = 16
PRO_FLOW_SHIFT = 3.0
PRO_MAX_DURATION_SEC = 5

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, flickering, static, noisy, "
    "overexposed, ugly, bad anatomy, watermark, text, jitter, "
    "oversaturated, deformed"
)

QUALITY_SUFFIX = (
    "cinematic lighting, smooth natural motion, sharp focus, "
    "high detail, professional color grading"
)


def enhance_prompt(prompt: str) -> str:
    """Light prompt boost for Wan T2V — skip if user already wrote quality tags."""
    p = prompt.strip()
    if not p:
        return p
    lower = p.lower()
    if any(
        k in lower
        for k in ("cinematic", "4k", "high quality", "smooth motion", "professional")
    ):
        return p
    return f"{p}, {QUALITY_SUFFIX}"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _is_pro_model(model_id: str) -> bool:
    return "14B" in model_id or "14b" in model_id.lower()


def _cuda_ready() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


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
            f"color=c=0x1a1a2e:s={PRO_WIDTH}x{PRO_HEIGHT}:d=2",
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
    _log(f"[wan_engine] smoke video → {output_path}")


def _check_model_disk(model_id: str) -> None:
    cache = os.environ.get("HF_HOME", "/tmp/huggingface")
    os.makedirs(cache, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / (1024**3)
    need_gb = 45.0 if _is_pro_model(model_id) else 15.0
    _log(f"[wan_engine] HF_HOME={cache} free={free_gb:.1f}GB need={need_gb:.0f}GB")
    if free_gb < need_gb:
        raise RuntimeError(
            f"Not enough disk at {cache}: {free_gb:.1f}GB free, need ~{need_gb:.0f}GB. "
            f"Set RunPod Container Disk to {'80' if _is_pro_model(model_id) else '50'}GB."
        )


def _configure_pipe(pipe, model_id: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU not available on worker")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    _log(f"[wan_engine] GPU {torch.cuda.get_device_name(0)} VRAM={vram_gb:.1f}GB")

    try:
        pipe.enable_attention_slicing("max")
    except Exception as e:
        _log(f"[wan_engine] attention_slicing skipped: {e}")

    if _is_pro_model(model_id):
        # 14B needs ~50GB+ VRAM full-GPU; RTX 4090 uses smart offload.
        if vram_gb >= 40:
            _log("[wan_engine] step 4/4 pipe.to(cuda) [14B datacenter GPU]")
            pipe.to("cuda")
        else:
            _log("[wan_engine] step 4/4 enable_model_cpu_offload [14B on 24GB]")
            pipe.enable_model_cpu_offload()
    elif vram_gb >= 14:
        _log("[wan_engine] step 4/4 pipe.to(cuda) [1.3B]")
        pipe.to("cuda")
    else:
        _log("[wan_engine] step 4/4 enable_model_cpu_offload [1.3B low VRAM]")
        pipe.enable_model_cpu_offload()


def _apply_scheduler(pipe, model_id: str) -> None:
    if not _is_pro_model(model_id):
        return
    try:
        from diffusers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=PRO_FLOW_SHIFT
        )
        _log(f"[wan_engine] scheduler flow_shift={PRO_FLOW_SHIFT}")
    except Exception as e:
        _log(f"[wan_engine] scheduler setup skipped: {e}")


def _load_pipeline(model_id: str):
    global _pipe, _model_id
    if _pipe is not None and _model_id == model_id:
        _log("[wan_engine] pipeline cache hit")
        return _pipe

    _check_model_disk(model_id)

    import torch

    _log(f"[wan_engine] step 1/4 import diffusers ({model_id})...")
    from diffusers import AutoencoderKLWan, WanPipeline

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log("[wan_engine] step 2/4 load VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    _log("[wan_engine] step 3/4 load WanPipeline (14B ~30GB first download)...")
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    _apply_scheduler(pipe, model_id)
    _configure_pipe(pipe, model_id)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    _pipe = pipe
    _model_id = model_id
    _log(f"[wan_engine] model ready ({model_id})")
    return _pipe


def warmup(model_id: str | None = None) -> None:
    if not _cuda_ready():
        _log("[wan_engine] warmup skipped — no CUDA")
        return
    mid = model_id or WAN_MODEL_ID
    _log(f"[wan_engine] warmup loading {mid}...")
    _load_pipeline(mid)
    _log("[wan_engine] warmup done — model hot")


def generate_video(
    prompt: str,
    duration_sec: int,
    output_path: Path,
    model_id: str = WAN_MODEL_ID,
) -> None:
    if not _cuda_ready():
        raise RuntimeError("CUDA/GPU not available on worker")

    from diffusers.utils import export_to_video

    import torch

    _log(f"[wan_engine] generate start model={model_id} prompt={prompt[:60]!r}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipe = _load_pipeline(model_id)
    capped = min(max(1, duration_sec), PRO_MAX_DURATION_SEC)
    enhanced = enhance_prompt(prompt)

    _log(
        f"[wan_engine] inference {PRO_FRAMES}f {PRO_WIDTH}x{PRO_HEIGHT} "
        f"{PRO_STEPS}steps (~{capped}s)..."
    )
    result = pipe(
        prompt=enhanced,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=PRO_FRAMES,
        width=PRO_WIDTH,
        height=PRO_HEIGHT,
        num_inference_steps=PRO_STEPS,
        guidance_scale=5.0,
    )

    _log("[wan_engine] export mp4...")
    export_to_video(result.frames[0], str(output_path), fps=PRO_FPS)
    sec = PRO_FRAMES / PRO_FPS
    _log(f"[wan_engine] done → {output_path} (~{sec:.1f}s @ {PRO_FPS}fps)")
