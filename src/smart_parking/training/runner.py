from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ultralytics import YOLO

from .dataset import generate_dataset_yaml


@dataclass
class SegmentationTrainingConfig:
    model: str | Path = "yolov8n-seg.pt"
    data_yaml: str | Path | None = None
    dataset_root: str | Path | None = None
    project: str | Path = "runs/train"
    name: str = "parking_slot_segmentation"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str | int | Sequence[int] | None = None
    workers: int = 8
    patience: int = 50
    seed: int = 42
    pretrained: bool = True
    cache: bool = False
    resume: bool = False
    plots: bool = True
    verbose: bool = True
    optimizer: str | None = None
    lr0: float | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)


def _resolve_data_yaml(config: SegmentationTrainingConfig) -> Path:
    if config.data_yaml is not None:
        return Path(config.data_yaml).resolve()
    if config.dataset_root is None:
        raise ValueError("Provide either data_yaml or dataset_root")
    return generate_dataset_yaml(config.dataset_root)


def train_segmentation_model(config: SegmentationTrainingConfig) -> Any:
    data_yaml = _resolve_data_yaml(config)
    model = YOLO(str(config.model))

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(config.epochs),
        "imgsz": int(config.imgsz),
        "batch": int(config.batch),
        "project": str(config.project),
        "name": config.name,
        "workers": int(config.workers),
        "patience": int(config.patience),
        "seed": int(config.seed),
        "pretrained": bool(config.pretrained),
        "cache": bool(config.cache),
        "resume": bool(config.resume),
        "plots": bool(config.plots),
        "verbose": bool(config.verbose),
    }
    if config.device is not None:
        train_kwargs["device"] = config.device
    if config.optimizer is not None:
        train_kwargs["optimizer"] = config.optimizer
    if config.lr0 is not None:
        train_kwargs["lr0"] = float(config.lr0)
    train_kwargs.update(config.extra_args)

    return model.train(**train_kwargs)
