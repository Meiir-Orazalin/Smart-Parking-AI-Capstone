from __future__ import annotations

from pathlib import Path

from .runner import SegmentationTrainingConfig, train_segmentation_model
from smart_parking.utils.paths import resolve_repo_path


def train_slot_model(
    *,
    data_yaml: str | Path,
    model: str = "yolov8n-seg.pt",
    project: str | Path = "runs/slot_training",
    name: str = "parking_slot_seg",
    imgsz: int = 1024,
    epochs: int = 100,
    batch: int = 4,
    device: str = "cpu",
    workers: int = 2,
) -> dict:
    data_yaml_path = resolve_repo_path(data_yaml)
    project_path = resolve_repo_path(project)
    config = SegmentationTrainingConfig(
        model=model,
        data_yaml=data_yaml_path,
        project=project_path,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        pretrained=True,
        cache=False,
        resume=False,
        plots=True,
        verbose=True,
    )
    result = train_segmentation_model(config)
    return {
        "data_yaml": str(data_yaml_path),
        "project": str(project_path),
        "name": name,
        "save_dir": str(getattr(result, "save_dir", project_path / name)),
    }
