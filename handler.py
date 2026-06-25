#!/usr/bin/env python3
"""RunPod — Wan 2.2 (5B standard / 14B I2V pro on A6000)."""
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
            import os

            tier = os.environ.get("WAN_MODEL_TIER", "standard")
            return {
                "ok": True,
                "worker": "v15-wan14b-pro",
                "model_tier": tier,
                "model": "Wan2.2-I2V-A14B" if tier == "pro" else "Wan2.2-TI2V-5B",
                "ads_lora": bool(__import__("os").environ.get("WAN_ADS_LORA_PATH", "").strip()),
            }

        prompt = str(inp.get("prompt", "")).strip()
        if not prompt:
            return {"error": "prompt required"}

        duration_sec = int(inp.get("duration_sec") or 8)
        watermark = bool(inp.get("watermark"))
        watermark_spec = inp.get("watermark_spec") or {}
        smoke_test = bool(inp.get("smoke_test"))
        generation_mode = str(inp.get("generation_mode") or "standard").strip().lower()
        ad_category = str(inp.get("ad_category") or "auto").strip().lower()
        ad_scene_style = str(inp.get("ad_scene_style") or "product").strip().lower()
        product_image_url = str(inp.get("product_image_url") or "").strip() or None
        model_tier = str(inp.get("model_tier") or "standard").strip().lower()

        with tempfile.TemporaryDirectory(prefix="runpod_video_") as tmp:
            tmp_dir = Path(tmp)
            raw_mp4 = tmp_dir / "raw.mp4"
            final_mp4 = tmp_dir / "final.mp4"

            if smoke_test:
                from wan_engine import generate_smoke_video

                generate_smoke_video(prompt, raw_mp4)
                model_name = "smoke"
            else:
                from wan_engine import generate_video, resolve_model_id

                model_id = resolve_model_id(
                    model_tier,
                    has_product_image=bool(product_image_url),
                    generation_mode=generation_mode,
                )
                generate_video(
                    prompt,
                    duration_sec,
                    raw_mp4,
                    model_id=model_id,
                    generation_mode=generation_mode,
                    ad_category=ad_category,
                    ad_scene_style=ad_scene_style,
                    product_image_url=product_image_url,
                )
                model_name = model_id
                if generation_mode == "ads":
                    model_name = f"{WAN_MODEL_ID}+super-ai-ads"

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

    print("[runpod] v15-wan14b-pro starting (lazy model load)...", flush=True)
    from wan_engine import warmup

    warmup()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
