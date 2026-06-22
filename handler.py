#!/usr/bin/env python3
"""RunPod Serverless handler — Wan 2.1 T2V 1.3B (4s clip)."""
from __future__ import annotations

import base64
import sys
import tempfile
import traceback
from pathlib import Path

import runpod


def handler(job: dict) -> dict:
    try:
        inp = job.get("input") or {}
        print(f"[handler] keys={list(inp.keys())}", flush=True)

        if inp.get("ping"):
            return {"ok": True, "worker": "v8j"}

        prompt = str(inp.get("prompt", "")).strip()
        if not prompt:
            return {"error": "prompt required"}

        duration_sec = int(inp.get("duration_sec") or 4)
        watermark = bool(inp.get("watermark"))
        watermark_spec = inp.get("watermark_spec") or {}
        smoke_test = bool(inp.get("smoke_test"))

        with tempfile.TemporaryDirectory(prefix="runpod_video_") as tmp:
            tmp_dir = Path(tmp)
            raw_mp4 = tmp_dir / "raw.mp4"
            final_mp4 = tmp_dir / "final.mp4"

            if smoke_test:
                from wan_engine import generate_smoke_video

                generate_smoke_video(prompt, raw_mp4)
                model_name = "smoke"
            else:
                from wan_engine import WAN_MODEL_ID, generate_video

                generate_video(prompt, duration_sec, raw_mp4, model_id=WAN_MODEL_ID)
                model_name = WAN_MODEL_ID

            out_path = final_mp4 if watermark else raw_mp4
            if watermark:
                from watermark_ffmpeg import burn_animated_watermark

                burn_animated_watermark(raw_mp4, final_mp4, watermark_spec)

            data = out_path.read_bytes()
            print(f"[handler] returning {len(data)} bytes", flush=True)
            return {
                "video_base64": base64.b64encode(data).decode("ascii"),
                "content_type": "video/mp4",
                "bytes": len(data),
                "model": model_name,
            }
    except Exception as e:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        return {"error": str(e)[:500]}


def main() -> None:
    import bootstrap  # noqa: F401

    print("[runpod] worker v8j starting...", flush=True)
    from wan_engine import warmup

    warmup()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
