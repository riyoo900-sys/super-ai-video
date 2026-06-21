"""Run before any diffusers import — env + torch patch + diffusers file patch."""
from __future__ import annotations

import os
import sys


def apply() -> None:
    os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        import torch

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
    except ImportError:
        pass

    try:
        import patch_diffusers

        patch_diffusers.main()
    except Exception as e:
        print(f"[bootstrap] patch_diffusers warning: {e}", file=sys.stderr)


apply()
