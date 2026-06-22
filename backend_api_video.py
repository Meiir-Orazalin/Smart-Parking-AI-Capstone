from __future__ import annotations

import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from smart_parking.api.video_backend import *  # noqa: F401,F403

