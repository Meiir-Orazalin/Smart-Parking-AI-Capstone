from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main(argv: list[str] | None = None) -> int:
    _bootstrap()
    from smart_parking.training.cli import main as training_main

    return training_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
