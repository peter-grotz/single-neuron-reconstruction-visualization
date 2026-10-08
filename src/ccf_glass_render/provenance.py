"""Provenance sidecar: what went in, so a figure can be regenerated."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .profile import RenderProfile
from .sources import Source


def _package_version() -> str:
    """Installed version, or the working-tree commit when running from source."""
    try:
        return version("ccf-glass-render")
    except PackageNotFoundError:
        pass
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True)
        return f"git+{sha.stdout.strip()}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write(path: Path, sources: list[Source], profile: RenderProfile,
          views: list[str], resolution_um: float, cells: list[str],
          outputs: list[Path], structures: list[str] | None = None) -> Path:
    """Write ``provenance.json`` beside the figures."""
    record = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "package": {"name": "ccf-glass-render", "version": _package_version()},
        "python": platform.python_version(),
        "profile": {"name": profile.name, "digest": profile.digest()},
        "atlas": {"template": "average_template_10", "resolution_um": resolution_um},
        "views": views,
        "structures": sorted(structures or []),
        "inputs": [{"uri": s.uri, "asset": s.asset_name, "subject": s.subject,
                    "swc_files": len(s.files)} for s in sources],
        "cells": sorted(cells),
        "outputs": [str(p.name) for p in outputs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
