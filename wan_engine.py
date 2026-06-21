"""Wan2.1-T2V-1.3B — singleton pipeline kept on GPU between jobs (fast path)."""
from __future__ import annotations

import os

# Native attention only — flash-attn custom_op breaks on PyTorch 2.4/2.5.
os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")


def _patch_torch_custom_ops() -> None:
    """Skip diffusers flash-attn @custom_op registration (infer_schema crash)."""
    try:
        import torch
    except ImportError:
        return
    if getattr(_patch_torch_custom_ops, "_done", False):
        return

    def custom_op_no_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    def register_fake_no_op(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    torch.library.custom_op = custom_op_no_op  # type: ignore[attr-defined]
    torch.library.register_fake = register_fake_no_op  # type: ignore[attr-defined]
    _patch_torch_custom_ops._done = True  # type: ignore[attr-defined]


_patch_torch_custom_ops()

import shutil
import subprocess
from pathlib import Path

_pipe = None
_model_id: str | None = None
_device: str | None = None

WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FAST_FRAMES = 17  # 4*k+1 minimum
FAST_STEPS = 18
FAST_WIDTH = 416
FAST_HEIGHT = 240
FAST_FPS = 6
FAST_MAX_DURATION_SEC = 4


def _ffmpeg_placeholder(prompt: str, duration_sec: int, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    safe = prompt.replace(":", r"\:").replace("'", r"\'")[:120]
    d = min(duration_sec, FAST_MAX_DURATION_SEC)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x1a1a2e:s={FAST_WIDTH}x{FAST_HEIGHT}:d={d}",
            "-vf",
            f"drawtext=text='{safe}':fontcolor=white:fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        timeout=60,
    )


def _cuda_ready() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_pipeline(model_id: str):
    global _pipe, _model_id, _device
    if _pipe is not None and _model_id == model_id:
        return _pipe

    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dtype = torch.bfloat16
    print(f"[wan_engine] loading {model_id} with CPU offload (T4 16GB safe)...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    pipe.enable_vae_slicing()
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass
    pipe.enable_model_cpu_offload()
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    _pipe = pipe
    _model_id = model_id
    _device = "cpu_offload"
    print("[wan_engine] model ready (CPU offload — fits T4)")
    return _pipe


def warmup(model_id: str | None = None) -> None:
    """Deferred — loading on first job avoids filling T4 VRAM at worker start."""
    if not _cuda_ready():
        print("[wan_engine] warmup skipped — no CUDA")
        return
    print("[wan_engine] warmup deferred until first job (VRAM safe)")


def generate_video(
    prompt: str,
    duration_sec: int,
    output_path: Path,
    model_id: str = WAN_MODEL_ID,
) -> None:
    if not _cuda_ready():
        raise RuntimeError(
            "CUDA/GPU not available on worker — cannot run Wan video model. "
            "Run videoWorkerSpeedPatch to install PyTorch CUDA on the GPU VM."
        )

    from diffusers.utils import export_to_video

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipe = _load_pipeline(model_id)
    # Short clip + few steps = sub-minute on warm T4.
    capped = min(max(1, duration_sec), FAST_MAX_DURATION_SEC)
    result = pipe(
        prompt=prompt,
        num_frames=FAST_FRAMES,
        width=FAST_WIDTH,
        height=FAST_HEIGHT,
        num_inference_steps=FAST_STEPS,
        guidance_scale=4.0,
    )
    export_to_video(result.frames[0], str(output_path), fps=FAST_FPS)
    print(f"[wan_engine] done → {output_path} (~{capped}s @ {FAST_FPS}fps, {FAST_FRAMES} frames)")
