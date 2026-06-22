"""Disable diffusers flash-attn custom_op (PyTorch infer_schema crash)."""
from __future__ import annotations

import pathlib
import re
import sys

MARKER = "# RUNPOD_PATCHED_v8"


def main() -> None:
    site = pathlib.Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    path = site / "diffusers/models/attention_dispatch.py"
    if not path.is_file():
        print(f"[patch_diffusers] skip — not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("[patch_diffusers] already patched v8")
        return

    # Disable torch custom_op binding in this module.
    text = text.replace(
        'if torch.__version__ >= "2.4.0":',
        f"if False:  {MARKER}",
        1,
    )

    # Remove flash-attn-3 registration block.
    text, n = re.subn(
        r'@_custom_op\("_diffusers_flash_attn_3::_flash_attn_forward"[^\n]*\n'
        r"def _wrapped_flash_attn_3\([\s\S]*?\n    return out, lse\n",
        f"# flash_attn_3 removed {MARKER}\n",
        text,
        count=1,
    )
    if n != 1:
        print("[patch_diffusers] WARNING — flash_attn_3 block not found, fallback")
        text = text.replace(
            '@_custom_op("_diffusers_flash_attn_3::_flash_attn_forward"',
            f"# {MARKER} disabled",
            1,
        )

    path.write_text(text, encoding="utf-8")

    # Fail build early if patch broke syntax.
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"[patch_diffusers] patched v8 → {path}")


if __name__ == "__main__":
    main()
