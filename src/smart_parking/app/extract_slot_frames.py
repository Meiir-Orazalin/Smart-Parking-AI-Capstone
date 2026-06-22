from __future__ import annotations

import argparse
import json

from smart_parking.training import extract_video_frames


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video for parking-slot labeling.")
    parser.add_argument("--video", required=True, help="Video path.")
    parser.add_argument("--output-dir", default="data/training/extracted_frames", help="Directory to write extracted frames.")
    parser.add_argument("--every-n-frames", type=int, default=60, help="Sample one frame every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to save. 0 means no limit.")
    parser.add_argument("--prefix", default="", help="Filename prefix.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = extract_video_frames(
        args.video,
        args.output_dir,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames or None,
        prefix=args.prefix or None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

