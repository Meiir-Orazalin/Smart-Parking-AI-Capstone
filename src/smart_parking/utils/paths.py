from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def default_video_path() -> Path:
    videos_dir = repo_root() / "assets" / "videos"
    candidates = (
        videos_dir / "ParkingVideo.MOV",
        videos_dir / "ParkingVideo.mov",
        videos_dir / "CarPark.mov",
        videos_dir / "CarPark.mp4",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_manual_slots_path() -> Path:
    return repo_root() / "data" / "slots" / "slots.json"


def default_slots_cache_path() -> Path:
    return repo_root() / "data" / "slots" / "auto_slots.json"


def default_slot_model_path() -> Path:
    return repo_root() / "assets" / "models" / "parking_slot_seg.pt"


def default_vehicle_model_path() -> Path:
    return repo_root() / "assets" / "models" / "yolov8m.pt"
