"""
TikTok-style animated watermark (corner drift) via ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH")
    return path


def burn_animated_watermark(
    input_mp4: Path,
    output_mp4: Path,
    spec: dict[str, Any],
) -> None:
    label = str(spec.get("labelText") or "SUPER AI VIDEO")
    opacity = float(spec.get("opacity") or 0.62)
    hold = float(spec.get("cornerHoldSeconds") or 2.5)
    width_frac = float(spec.get("widthFraction") or 0.22)

    safe_label = label.replace(":", r"\:").replace("'", r"\'")
    fontsize_expr = f"min(w\\,h)*{width_frac * 0.12:.4f}"
    cycle = hold * 4
    x_expr = (
        f"if(lt(mod(t\\,{cycle})\\,{hold})\\,w*0.04\\,"
        f"if(lt(mod(t\\,{cycle})\\,{hold*2})\\,w-text_w-w*0.04\\,"
        f"if(lt(mod(t\\,{cycle})\\,{hold*3})\\,w*0.04\\,w-text_w-w*0.04)))"
    )
    y_expr = (
        f"if(lt(mod(t\\,{cycle})\\,{hold})\\,h*0.08\\,"
        f"if(lt(mod(t\\,{cycle})\\,{hold*2})\\,h*0.08\\,"
        f"if(lt(mod(t\\,{cycle})\\,{hold*3})\\,h-text_h-h*0.08\\,h-text_h-h*0.08)))"
    )

    vf = (
        f"drawtext=text='{safe_label}':fontcolor=white@{opacity}:fontsize={fontsize_expr}:"
        f"box=1:boxcolor=black@0.45:boxborderw=8:x={x_expr}:y={y_expr}"
    )

    cmd = [
        _ffmpeg(),
        "-y",
        "-i",
        str(input_mp4),
        "-vf",
        vf,
        "-c:a",
        "copy",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True, timeout=300)
