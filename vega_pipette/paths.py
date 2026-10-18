"""Repository locations, from environment variables with defaults relative to the repository.

  VEGA_HOME  the repository root (default: the directory above vega_pipette/, which is installed editable)
  VEGA_RUNS  where runs, videos and caches go (default: <VEGA_HOME>/runs)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ.get("VEGA_HOME") or Path(__file__).resolve().parents[1]).resolve()
ASSETS = REPO / "assets"
RUNS = Path(os.environ.get("VEGA_RUNS") or REPO / "runs").resolve()
if not ASSETS.is_dir():
    raise RuntimeError(f"no assets/ under {REPO}: install vega_pipette editable from the repository (pip install -e .), "
                       "or set VEGA_HOME to the repository")
