#!/usr/bin/env python3
"""Entrypoint: patch diffusers, verify import, start RunPod handler."""
import bootstrap  # noqa: F401


def _verify() -> None:
    from diffusers import AutoencoderKLWan, WanPipeline  # noqa: F401
    import diffusers

    print(f"[start] wan import OK diffusers={diffusers.__version__}")


if __name__ == "__main__":
    _verify()
    import handler

    handler.main()
