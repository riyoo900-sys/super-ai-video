"""Disable diffusers flash-attn custom_op (PyTorch infer_schema crash on 2.5.x)."""
from __future__ import annotations

import pathlib
import re
import sys

MARKER = "# RUNPOD_PATCHED_v5"


def main() -> None:
    site = pathlib.Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    path = site / "diffusers/models/attention_dispatch.py"
    if not path.is_file():
        print(f"[patch_diffusers] skip — not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("[patch_diffusers] already patched v5")
        return

    # 1) Never bind real torch custom_op in this module.
    text = text.replace(
        'if torch.__version__ >= "2.4.0":',
        f"if False:  {MARKER}",
        1,
    )

    # 2) Remove flash-attn-3 custom op registration (crashes infer_schema).
    text, n = re.subn(
        r'@_custom_op\("_diffusers_flash_attn_3::_flash_attn_forward"[^\n]*\n'
        r"def _wrapped_flash_attn_3\([\s\S]*?\n    return out, lse\n",
        f"# flash_attn_3 custom_op removed {MARKER}\n",
        text,
        count=1,
    )
    if n != 1:
        print("[patch_diffusers] WARNING — flash_attn block not removed, using fallback")
        text = text.replace(
            '@_custom_op("_diffusers_flash_attn_3::_flash_attn_forward"',
            f"# {MARKER} @_custom_op disabled",
            1,
        )

    path.write_text(text, encoding="utf-8")
    print(f"[patch_diffusers] patched v5 → {path}")


if __name__ == "__main__":
    main()
