#!/usr/bin/env python3
"""RunPod Serverless handler — Wan 2.1 T2V 1.3B (4s clip)."""
from __future__ import annotations

import bootstrap  # noqa: F401 — patch diffusers before any model import

import base64
import tempfile
import traceback
from pathlib import Path

import runpod

from wan_engine import WAN_MODEL_ID, generate_video, warmup
from watermark_ffmpeg import burn_animated_watermark


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    prompt = str(inp.get("prompt", "")).strip()
    if not prompt:
        return {"error": "prompt required"}

    duration_sec = int(inp.get("duration_sec") or 4)
    watermark = bool(inp.get("watermark"))
    watermark_spec = inp.get("watermark_spec") or {}

    try:
        with tempfile.TemporaryDirectory(prefix="runpod_video_") as tmp:
            tmp_dir = Path(tmp)
            raw_mp4 = tmp_dir / "raw.mp4"
            final_mp4 = tmp_dir / "final.mp4"

            generate_video(prompt, duration_sec, raw_mp4, model_id=WAN_MODEL_ID)

            out_path = final_mp4 if watermark else raw_mp4
            if watermark:
                burn_animated_watermark(raw_mp4, final_mp4, watermark_spec)

            data = out_path.read_bytes()
            return {
                "video_base64": base64.b64encode(data).decode("ascii"),
                "content_type": "video/mp4",
                "bytes": len(data),
                "model": WAN_MODEL_ID,
            }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)[:500]}


def main() -> None:
    import diffusers

    print(f"[runpod] worker v8 diffusers={diffusers.__version__}")
    warmup()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
