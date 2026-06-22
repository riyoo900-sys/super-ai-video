"""Wan2.1-T2V-1.3B — singleton pipeline kept on GPU between jobs."""
from __future__ import annotations

import bootstrap  # noqa: F401 — patch diffusers before import

import os
import shutil
import subprocess
from pathlib import Path

_pipe = None
_model_id: str | None = None

WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FAST_FRAMES = 13
FAST_STEPS = 12
FAST_WIDTH = 416
FAST_HEIGHT = 240
FAST_FPS = 6
FAST_MAX_DURATION_SEC = 4


def _log(msg: str) -> None:
    print(msg, flush=True)


def _cuda_ready() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def generate_smoke_video(prompt: str, output_path: Path) -> None:
    """Tiny MP4 without GPU model — verifies RunPod handler path."""
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
            f"color=c=0x1a1a2e:s={FAST_WIDTH}x{FAST_HEIGHT}:d=2",
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


def _check_model_disk(min_gb: float = 15.0) -> None:
    cache = os.environ.get("HF_HOME", "/tmp/huggingface")
    os.makedirs(cache, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / (1024**3)
    _log(f"[wan_engine] HF_HOME={cache} free={free_gb:.1f}GB")
    if free_gb < min_gb:
        raise RuntimeError(
            f"Not enough disk space at {cache}: {free_gb:.1f}GB free, need ~{min_gb}GB. "
            "In RunPod endpoint settings set Container Disk to at least 50GB, then New Build."
        )


def _configure_pipe(pipe) -> None:
    """Put weights on GPU when VRAM allows — CPU-only inference times out on RunPod."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU not available on worker")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    _log(f"[wan_engine] GPU {torch.cuda.get_device_name(0)} VRAM={vram_gb:.1f}GB")

    try:
        pipe.enable_attention_slicing("max")
    except Exception as e:
        _log(f"[wan_engine] attention_slicing skipped: {e}")

    # Wan 1.3B fits on 16–24GB; must use GPU — attention_slicing alone leaves model on CPU.
    if vram_gb >= 14:
        _log("[wan_engine] step 4/4 pipe.to(cuda)")
        pipe.to("cuda")
    else:
        _log("[wan_engine] step 4/4 enable_model_cpu_offload (low VRAM)")
        pipe.enable_model_cpu_offload()


def _load_pipeline(model_id: str):
    global _pipe, _model_id
    if _pipe is not None and _model_id == model_id:
        _log("[wan_engine] pipeline cache hit")
        return _pipe

    _check_model_disk()

    import torch

    _log("[wan_engine] step 1/4 import diffusers...")
    from diffusers import AutoencoderKLWan, WanPipeline

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log(f"[wan_engine] step 2/4 load VAE ({model_id})...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    _log("[wan_engine] step 3/4 load WanPipeline...")
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    _configure_pipe(pipe)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    _pipe = pipe
    _model_id = model_id
    _log("[wan_engine] model ready on GPU")
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

    _log(f"[wan_engine] generate start prompt={prompt[:60]!r}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipe = _load_pipeline(model_id)
    capped = min(max(1, duration_sec), FAST_MAX_DURATION_SEC)

    _log(f"[wan_engine] inference {FAST_FRAMES} frames {FAST_WIDTH}x{FAST_HEIGHT}...")
    result = pipe(
        prompt=prompt,
        num_frames=FAST_FRAMES,
        width=FAST_WIDTH,
        height=FAST_HEIGHT,
        num_inference_steps=FAST_STEPS,
        guidance_scale=4.0,
    )

    _log("[wan_engine] export mp4...")
    export_to_video(result.frames[0], str(output_path), fps=FAST_FPS)
    _log(f"[wan_engine] done → {output_path} (~{capped}s @ {FAST_FPS}fps)")
