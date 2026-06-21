"""Disable diffusers flash-attn custom_op registration (PyTorch infer_schema crash)."""
from __future__ import annotations

import pathlib
import re
import sys


def main() -> None:
    site = pathlib.Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    path = site / "diffusers/models/attention_dispatch.py"
    if not path.is_file():
        print(f"[patch_diffusers] skip — not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    marker = "# RUNPOD_PATCHED"
    if marker in text:
        print("[patch_diffusers] already patched")
        return

    pattern = (
        r'if torch\.__version__ >= "2\.4\.0":\s*\n'
        r"\s*_custom_op = torch\.library\.custom_op\s*\n"
        r"\s*_register_fake = torch\.library\.register_fake"
    )
    replacement = (
        f"if False:  {marker}\n"
        "    _custom_op = torch.library.custom_op\n"
        "    _register_fake = torch.library.register_fake"
    )
    new_text, n = re.subn(pattern, replacement, text, count=1)
    if n != 1:
        print("[patch_diffusers] WARNING — pattern not found, trying fallback")
        new_text = text.replace(
            '_custom_op = torch.library.custom_op',
            f'_custom_op = torch.library.custom_op  {marker}',
            1,
        )
        # Force no-op block by replacing the if condition
        new_text = new_text.replace(
            'if torch.__version__ >= "2.4.0":',
            f"if False:  {marker}",
            1,
        )

    path.write_text(new_text, encoding="utf-8")
    print(f"[patch_diffusers] patched {path}")


if __name__ == "__main__":
    main()
