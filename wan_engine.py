"""Wan T2V — RTX 4090 optimized: 1.3B full-GPU @ 480p (speed + clarity)."""
from __future__ import annotations

import bootstrap  # noqa: F401 — patch diffusers before import

import os
import shutil
import subprocess
from pathlib import Path

_pipe = None
_model_id: str | None = None

# Default on 24GB (4090): 1.3B full GPU — faster and sharper than 14B + CPU offload.
WAN_MODEL_ID = os.environ.get(
    "WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
)
WAN_MODEL_PRO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# 480p — tuned for clarity on 1.3B (Wan: num_frames = 4n+1).
LITE_FRAMES = 41
LITE_STEPS = 32
LITE_WIDTH = 832
LITE_HEIGHT = 480
LITE_FPS = 16
LITE_MAX_DURATION_SEC = 5

PRO_FRAMES = 49
PRO_STEPS = 30
PRO_WIDTH = 832
PRO_HEIGHT = 480
PRO_FPS = 16
PRO_FLOW_SHIFT = 3.0
PRO_MAX_DURATION_SEC = 5

NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, pixelated, distorted, flickering, static, "
    "noisy, grainy, overexposed, ugly, bad anatomy, watermark, text, jitter, "
    "oversaturated, deformed, cartoon, painting, illustration, artificial"
)

REALISM_SUFFIX = (
    "photorealistic, natural smooth motion, sharp focus, realistic lighting, "
    "high detail, lifelike, cinematic 4K look, real world footage"
)


def enhance_prompt(prompt: str) -> str:
    p = prompt.strip()
    if not p:
        return p
    lower = p.lower()
    if any(
        k in lower
        for k in (
            "photorealistic",
            "realistic",
            "4k",
            "high quality",
            "lifelike",
            "cinematic",
            "sharp focus",
        )
    ):
        return p
    return f"{p}, {REALISM_SUFFIX}"


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


def _apply_lite_optimizations(pipe) -> None:
    """1.3B on 24GB: light slicing only — keep max sharpness."""
    try:
        pipe.enable_attention_slicing("auto")
    except Exception as e:
        _log(f"[wan_engine] attention_slicing skipped: {e}")


def _apply_pro_optimizations(pipe) -> None:
    try:
        pipe.enable_attention_slicing("max")
    except Exception as e:
        _log(f"[wan_engine] attention_slicing skipped: {e}")
    try:
        pipe.enable_vae_slicing()
    except Exception as e:
        _log(f"[wan_engine] vae_slicing skipped: {e}")
    try:
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.enable_tiling()
    except Exception as e:
        _log(f"[wan_engine] vae_tiling skipped: {e}")


def _inference_profile(model_id: str) -> tuple[int, int, int, int, int]:
    """frames, width, height, steps, fps."""
    if _is_pro_model(model_id):
        vram_gb = _gpu_vram_gb()
        if vram_gb >= 48:
            return PRO_FRAMES, PRO_WIDTH, PRO_HEIGHT, PRO_STEPS, PRO_FPS
        _log("[wan_engine] 14B 24GB profile: 33f 28steps (offload)")
        return 33, PRO_WIDTH, PRO_HEIGHT, 28, PRO_FPS

    _log(
        f"[wan_engine] 1.3B 480p profile: {LITE_FRAMES}f {LITE_STEPS}steps "
        f"@ {LITE_WIDTH}x{LITE_HEIGHT}"
    )
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
            f"color=c=0x1a1a2e:s={LITE_WIDTH}x{LITE_HEIGHT}:d=2",
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
    need_gb = 45.0 if _is_pro_model(model_id) else 12.0
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

    vram_gb = _gpu_vram_gb()
    _log(f"[wan_engine] GPU {torch.cuda.get_device_name(0)} VRAM={vram_gb:.1f}GB")

    _clear_cuda()

    if _is_pro_model(model_id):
        try:
            pipe.to("cpu")
        except Exception:
            pass
        _clear_cuda()
        _apply_pro_optimizations(pipe)
        if vram_gb >= 48:
            _log("[wan_engine] step 4/4 pipe.to(cuda) [14B 48GB+]")
            pipe.to("cuda")
        else:
            _log("[wan_engine] step 4/4 sequential_cpu_offload [14B 24GB]")
            pipe.enable_sequential_cpu_offload()
    else:
        _apply_lite_optimizations(pipe)
        if vram_gb >= 10:
            _log("[wan_engine] step 4/4 pipe.to(cuda) [1.3B full GPU — 4090]")
            pipe.to("cuda")
        else:
            _log("[wan_engine] step 4/4 model_cpu_offload [1.3B low VRAM]")
            pipe.enable_model_cpu_offload()

    _clear_cuda()
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        _log(f"[wan_engine] CUDA allocated after configure: {alloc_gb:.2f}GB")


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

    _clear_cuda()

    _log("[wan_engine] step 2/4 load VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    _log("[wan_engine] step 3/4 load WanPipeline...")
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

    _log(f"[wan_engine] generate start model={model_id} prompt={prompt[:60]!r}")
    _clear_cuda()

    pipe = _load_pipeline(model_id)
    max_dur = PRO_MAX_DURATION_SEC if _is_pro_model(model_id) else LITE_MAX_DURATION_SEC
    capped = min(max(1, duration_sec), max_dur)
    enhanced = enhance_prompt(prompt)
    num_frames, width, height, steps, fps = _inference_profile(model_id)

    _clear_cuda()
    _log(
        f"[wan_engine] inference {num_frames}f {width}x{height} "
        f"{steps}steps (~{capped}s)..."
    )
    result = pipe(
        prompt=enhanced,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=num_frames,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=6.0,
    )

    _log("[wan_engine] export mp4...")
    export_to_video(result.frames[0], str(output_path), fps=fps)
    sec = num_frames / fps
    _log(f"[wan_engine] done → {output_path} (~{sec:.1f}s @ {fps}fps)")
