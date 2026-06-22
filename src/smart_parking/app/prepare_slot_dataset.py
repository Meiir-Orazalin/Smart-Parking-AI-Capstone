from __future__ import annotations

import argparse
import json

from smart_parking.training import prepare_yolo_seg_dataset
from smart_parking.utils.paths import repo_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO segmentation dataset for parking-slot training.")
    parser.add_argument("--manifest", default="data/training/parking_cropped_manifest.json", help="Dataset manifest JSON.")
    parser.add_argument("--output-dir", default="data/training/parking_cropped_dataset", help="Output YOLO dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio when no split is specified.")
    parser.add_argument("--symlink-images", action="store_true", help="Symlink images instead of copying them.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = prepare_yolo_seg_dataset(
        args.manifest,
        args.output_dir,
        val_ratio=args.val_ratio,
        copy_images=not args.symlink_images,
    )
    print(json.dumps(summary, indent=2))
    print(f"Dataset ready under {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
