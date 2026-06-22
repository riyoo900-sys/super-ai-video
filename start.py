#!/usr/bin/env python3
"""Entrypoint: patch diffusers, verify import, start RunPod handler."""
from __future__ import annotations

import sys
import traceback


def _fatal(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        import bootstrap  # noqa: F401

        from diffusers import AutoencoderKLWan, WanPipeline  # noqa: F401
        import diffusers

        print(f"[start] wan import OK diffusers={diffusers.__version__}", flush=True)

        import handler

        handler.main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
