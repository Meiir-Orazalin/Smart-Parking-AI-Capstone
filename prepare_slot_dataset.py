#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> int:
    _bootstrap()
    from smart_parking.app.prepare_slot_dataset import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())

