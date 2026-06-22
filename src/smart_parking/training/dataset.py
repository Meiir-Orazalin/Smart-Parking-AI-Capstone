from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2

from smart_parking.slots import canonicalize_slots, load_slot_file, select_view_payload
from smart_parking.slots.geometry import coerce_image_shape, normalize_slots, scale_slots

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    annotation_path: Path | None = None
    view_id: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[DatasetSample, ...]
    val: tuple[DatasetSample, ...]


@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    dataset_yaml: Path
    train_count: int
    val_count: int
    class_names: tuple[str, ...]


def discover_image_paths(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image source not found: {path}")
    return [
        candidate
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _read_image_shape(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image.shape[:2]


def _is_view_cache(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("version") == 1 and isinstance(payload.get("views"), list)


def _resolve_annotation_payload(
    annotation_path: str | Path,
    *,
    view_id: str | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
    payload = load_slot_file(annotation_path)
    if _is_view_cache(payload):
        view = select_view_payload(payload, view_id=view_id)
        if view is None:
            raise ValueError(f"No matching view found in cache: {annotation_path}")
        slots = list(view.get("slots", []))
        image_shape = coerce_image_shape(view.get("image_shape")) or coerce_image_shape(payload.get("image_shape"))
        return slots, image_shape

    if isinstance(payload, dict):
        slots = list(payload.get("slots", []))
        image_shape = coerce_image_shape(payload.get("image_shape"))
        return slots, image_shape

    if isinstance(payload, list):
        return list(payload), None

    raise ValueError(f"Unsupported annotation payload: {annotation_path}")


def slots_from_annotation_source(
    annotation_path: str | Path,
    *,
    image_path: str | Path | None = None,
    view_id: str | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
    slots, source_shape = _resolve_annotation_payload(annotation_path, view_id=view_id)
    if image_path is None:
        return canonicalize_slots(slots, image_shape=source_shape), source_shape

    target_shape = _read_image_shape(Path(image_path))
    if source_shape and source_shape != target_shape:
        scale_x = target_shape[1] / source_shape[1] if source_shape[1] else 1.0
        scale_y = target_shape[0] / source_shape[0] if source_shape[0] else 1.0
        slots = scale_slots(slots, scale_x, scale_y)
    slots = normalize_slots(slots, image_shape=target_shape)
    return slots, target_shape


def _label_lines_from_slots(
    slots: Sequence[dict[str, Any]],
    image_shape: tuple[int, int],
    *,
    class_id: int = 0,
) -> list[str]:
    height, width = image_shape
    lines: list[str] = []
    for slot in slots:
        points = slot.get("points", [])
        if len(points) < 3:
            continue
        normalized_points: list[str] = []
        for point in points:
            x = min(1.0, max(0.0, float(point[0]) / max(1, width)))
            y = min(1.0, max(0.0, float(point[1]) / max(1, height)))
            normalized_points.append(f"{x:.6f} {y:.6f}")
        if normalized_points:
            lines.append(f"{class_id} " + " ".join(normalized_points))
    return lines


def write_yolo_seg_label(
    label_path: str | Path,
    slots: Sequence[dict[str, Any]],
    image_shape: tuple[int, int],
    *,
    class_id: int = 0,
) -> Path:
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_lines = _label_lines_from_slots(slots, image_shape, class_id=class_id)
    label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return label_path


def _copy_or_link_image(source: Path, target: Path, *, copy_images: bool) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(source, target)
        return target
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())
    return target


def split_samples(samples: Sequence[DatasetSample], *, train_ratio: float = 0.8, seed: int = 42) -> DatasetSplit:
    if not samples:
        return DatasetSplit(train=tuple(), val=tuple())
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    if len(ordered) == 1:
        return DatasetSplit(train=(ordered[0],), val=tuple())
    cutoff = int(round(len(ordered) * max(0.0, min(train_ratio, 0.95))))
    cutoff = max(1, min(len(ordered) - 1, cutoff))
    return DatasetSplit(train=tuple(ordered[:cutoff]), val=tuple(ordered[cutoff:]))


def generate_dataset_yaml(
    dataset_root: str | Path,
    *,
    output_path: str | Path | None = None,
    class_names: Sequence[str] = ("slot",),
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path) if output_path is not None else dataset_root / "data.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for name in class_names:
        lines.append(f"  - {name}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def prepare_sample_labels(
    image_path: str | Path,
    annotation_source: str | Path,
    label_path: str | Path,
    *,
    class_id: int = 0,
    view_id: str | None = None,
) -> Path:
    image_path = Path(image_path)
    slots, image_shape = slots_from_annotation_source(annotation_source, image_path=image_path, view_id=view_id)
    if image_shape is None:
        image_shape = _read_image_shape(image_path)
    return write_yolo_seg_label(label_path, slots, image_shape, class_id=class_id)


def _resolve_slots_for_image(
    image_path: Path,
    slots_source: Path,
    *,
    view_id: str | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
    if slots_source.is_dir():
        candidates = [
            slots_source / f"{image_path.stem}.json",
            slots_source / f"{image_path.stem}.slots.json",
            slots_source / f"{image_path.stem}.slot.json",
        ]
        annotation_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if annotation_path is None:
            raise FileNotFoundError(f"No annotation file found for image: {image_path.name}")
        return slots_from_annotation_source(annotation_path, image_path=image_path, view_id=view_id)
    return slots_from_annotation_source(slots_source, image_path=image_path, view_id=view_id)


def prepare_dataset(
    images_source: str | Path,
    slots_source: str | Path,
    output_root: str | Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    view_id: str | None = None,
    class_names: Sequence[str] = ("slot",),
    class_id: int = 0,
    copy_images: bool = True,
) -> PreparedDataset:
    image_paths = discover_image_paths(images_source)
    if len(image_paths) < 2:
        raise ValueError("Need at least two images to create a train/val dataset.")

    split = split_samples(
        tuple(DatasetSample(image_path=image_path) for image_path in image_paths),
        train_ratio=train_ratio,
        seed=seed,
    )
    train_set = {sample.image_path for sample in split.train}

    output_root = Path(output_root).resolve()
    image_dirs = {
        "train": output_root / "images" / "train",
        "val": output_root / "images" / "val",
    }
    label_dirs = {
        "train": output_root / "labels" / "train",
        "val": output_root / "labels" / "val",
    }
    for folder in (*image_dirs.values(), *label_dirs.values()):
        folder.mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0
    for image_path in image_paths:
        slots, image_shape = _resolve_slots_for_image(Path(image_path), Path(slots_source), view_id=view_id)
        split_name = "train" if Path(image_path) in train_set else "val"
        if image_shape is None:
            image_shape = _read_image_shape(Path(image_path))

        image_target = image_dirs[split_name] / Path(image_path).name
        label_target = label_dirs[split_name] / f"{Path(image_path).stem}.txt"
        _copy_or_link_image(Path(image_path), image_target, copy_images=copy_images)
        write_yolo_seg_label(label_target, slots, image_shape, class_id=class_id)

        if split_name == "train":
            train_count += 1
        else:
            val_count += 1

    dataset_yaml = generate_dataset_yaml(output_root, class_names=class_names)
    return PreparedDataset(
        root=output_root,
        dataset_yaml=dataset_yaml,
        train_count=train_count,
        val_count=val_count,
        class_names=tuple(class_names),
    )


def prepare_yolo_seg_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    val_ratio: float = 0.2,
    copy_images: bool = True,
    class_name: str = "slot",
) -> dict[str, Any]:
    """Backward-compatible manifest-based dataset builder.

    Expected manifest structure:
    - a list of items, or
    - a dict with an ``items`` list
    where each item contains ``image`` and either ``slots`` or ``slots_path``.
    """

    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        items = data.get("items", [])
    else:
        items = data
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset manifest is empty")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_items: list[dict[str, Any]] = []
    for item in items:
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            manifest_relative = (manifest_path.parent / image_path).resolve()
            image_path = manifest_relative if manifest_relative.exists() else (Path.cwd() / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_shape = _read_image_shape(image_path)
        if "slots" in item and isinstance(item["slots"], list):
            slots = canonicalize_slots(item["slots"], image_shape=image_shape)
        else:
            slots_source = item.get("slots_path") or item.get("slots")
            if not slots_source:
                raise ValueError(f"Missing slots for manifest item: {item}")
            slots_source = Path(slots_source)
            if not slots_source.is_absolute():
                manifest_relative = (manifest_path.parent / slots_source).resolve()
                slots_source = manifest_relative if manifest_relative.exists() else (Path.cwd() / slots_source).resolve()
            slots, _ = slots_from_annotation_source(slots_source, image_path=image_path, view_id=item.get("view_id"))
        if not slots:
            continue
        normalized_items.append(
            {
                "image_path": image_path,
                "image_shape": image_shape,
                "slots": slots,
                "split": item.get("split"),
                "name": item.get("name") or image_path.stem,
            }
        )

    if not normalized_items:
        raise ValueError("No usable labeled items were found in the manifest")

    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"
    for path in (images_train, images_val, labels_train, labels_val):
        path.mkdir(parents=True, exist_ok=True)

    auto_val_count = max(1, int(round(len(normalized_items) * max(0.0, min(val_ratio, 0.9))))) if len(normalized_items) > 1 else 1
    val_names = {item["name"] for item in normalized_items[-auto_val_count:]}

    train_count = 0
    val_count = 0
    for item in normalized_items:
        split = item["split"] or ("val" if item["name"] in val_names else "train")
        if split not in {"train", "val"}:
            split = "train"
        image_target = (images_val if split == "val" else images_train) / item["image_path"].name
        label_target = (labels_val if split == "val" else labels_train) / f"{item['name']}.txt"
        _copy_or_link_image(item["image_path"], image_target, copy_images=copy_images)
        write_yolo_seg_label(label_target, item["slots"], item["image_shape"], class_id=0)
        if split == "val":
            val_count += 1
        else:
            train_count += 1

    dataset_yaml = generate_dataset_yaml(output_dir, class_names=(class_name,))
    return {
        "output_dir": str(output_dir),
        "dataset_yaml": str(dataset_yaml),
        "train_images": train_count,
        "val_images": val_count,
        "total_items": len(normalized_items),
    }
