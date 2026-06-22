from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2


@dataclass(frozen=True)
class FrameExtractionResult:
    video_path: Path
    output_dir: Path
    frames: tuple[Path, ...]
    total_read: int
    total_saved: int


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    every_n_frames: int = 30,
    max_frames: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    prefix: str | None = None,
    image_ext: str = ".jpg",
) -> FrameExtractionResult:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if every_n_frames < 1:
        raise ValueError("every_n_frames must be >= 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
    frame_index = max(0, int(start_frame))
    total_read = 0
    total_saved = 0
    saved_paths: list[Path] = []
    stem = prefix or video_path.stem

    while True:
        if end_frame is not None and frame_index > int(end_frame):
            break
        if max_frames is not None and total_saved >= int(max_frames):
            break

        ok, frame = capture.read()
        if not ok:
            break

        total_read += 1
        should_save = (frame_index - start_frame) % every_n_frames == 0
        if should_save:
            filename = f"{stem}_{frame_index:06d}{image_ext}"
            target_path = output_dir / filename
            cv2.imwrite(str(target_path), frame)
            saved_paths.append(target_path)
            total_saved += 1

        frame_index += 1

    capture.release()
    return FrameExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        frames=tuple(saved_paths),
        total_read=total_read,
        total_saved=total_saved,
    )
