"""Wan2.1-T2V-1.3B — singleton pipeline kept on GPU between jobs."""
from __future__ import annotations

import bootstrap  # noqa: F401 — patch diffusers before import

import shutil
import subprocess
from pathlib import Path

_pipe = None
_model_id: str | None = None

WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FAST_FRAMES = 17
FAST_STEPS = 18
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


def _load_pipeline(model_id: str):
    global _pipe, _model_id
    if _pipe is not None and _model_id == model_id:
        return _pipe

    import torch

    _log("[wan_engine] step 1/4 import diffusers...")
    from diffusers import AutoencoderKLWan, WanPipeline

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log(f"[wan_engine] step 2/4 load VAE ({model_id})...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    _log("[wan_engine] step 3/4 load WanPipeline (downloads ~10GB first time)...")
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    pipe.enable_vae_slicing()
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass

    # Never pipe.to("cuda") — UMT5-XXL + Wan OOMs even on 48GB.
    _log("[wan_engine] step 4/4 enable_sequential_cpu_offload...")
    pipe.enable_sequential_cpu_offload()

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    _pipe = pipe
    _model_id = model_id
    _log("[wan_engine] model ready")
    return _pipe


def warmup(model_id: str | None = None) -> None:
    if not _cuda_ready():
        _log("[wan_engine] warmup skipped — no CUDA")
        return
    _log("[wan_engine] warmup deferred until first job")


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
