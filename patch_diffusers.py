"""Disable diffusers flash-attn custom_op (PyTorch infer_schema crash)."""
from __future__ import annotations

import pathlib
import re
import sys

MARKER = "# RUNPOD_PATCHED_v6"


def main() -> None:
    site = pathlib.Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    path = site / "diffusers/models/attention_dispatch.py"
    if not path.is_file():
        print(f"[patch_diffusers] skip — not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("[patch_diffusers] already patched v6")
        return

    # Never register torch custom ops in this module.
    text = text.replace(
        'if torch.__version__ >= "2.4.0":',
        f"if False:  {MARKER}",
    )
    text = text.replace(
        "@torch.library.custom_op",
        f"# {MARKER} @torch.library.custom_op",
    )
    text = text.replace(
        "@_custom_op(",
        f"# {MARKER} @_custom_op(",
    )

    # Remove flash-attn-3 wrapper if still present.
    text, n = re.subn(
        r'def _wrapped_flash_attn_3\([\s\S]*?\n    return out, lse\n',
        f"def _wrapped_flash_attn_3(*args, **kwargs):  {MARKER}\n    raise RuntimeError('flash_attn_3 disabled')\n",
        text,
        count=1,
    )
    if n:
        print("[patch_diffusers] replaced _wrapped_flash_attn_3")

    path.write_text(text, encoding="utf-8")
    print(f"[patch_diffusers] patched v6 → {path}")


if __name__ == "__main__":
    main()
