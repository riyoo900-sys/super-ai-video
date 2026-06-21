"""Run before any diffusers import — env + torch patch + diffusers file patch."""
from __future__ import annotations

import os
import sys


def _noop_custom_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
    def wrap(func):
        return func

    return wrap if fn is None else fn


def _noop_register_fake(op, fn=None, /, *, lib=None, _stacklevel=1):
    def wrap(func):
        return func

    return wrap if fn is None else fn


def apply() -> None:
    os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        import torch

        torch.library.custom_op = _noop_custom_op  # type: ignore[attr-defined]
        torch.library.register_fake = _noop_register_fake  # type: ignore[attr-defined]
        try:
            import torch._library.custom_ops as custom_ops

            custom_ops.custom_op = _noop_custom_op  # type: ignore[attr-defined]
        except ImportError:
            pass
    except ImportError:
        pass

    try:
        import patch_diffusers

        patch_diffusers.main()
    except Exception as e:
        print(f"[bootstrap] patch_diffusers warning: {e}", file=sys.stderr)


apply()
