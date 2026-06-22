from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

Point = tuple[int, int]
BBox = tuple[int, int, int, int]
ImageShape = tuple[int, int]


def _coerce_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def coerce_image_shape(image_shape: Sequence[Any] | None) -> ImageShape | None:
    if not image_shape or len(image_shape) < 2:
        return None
    height = _coerce_int(image_shape[0])
    width = _coerce_int(image_shape[1])
    if height is None or width is None or height <= 0 or width <= 0:
        return None
    return height, width


def clamp_bbox(bbox: Sequence[Any], image_shape: ImageShape | None = None) -> BBox | None:
    if bbox is None or len(bbox) < 4:
        return None
    x1 = _coerce_int(bbox[0])
    y1 = _coerce_int(bbox[1])
    x2 = _coerce_int(bbox[2])
    y2 = _coerce_int(bbox[3])
    if None in (x1, y1, x2, y2):
        return None

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if image_shape is not None:
        height, width = image_shape
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

    if x1 == x2 or y1 == y2:
        return None
    return x1, y1, x2, y2


def points_from_bbox(bbox: Sequence[Any]) -> list[Point]:
    normalized = clamp_bbox(bbox)
    if normalized is None:
        return []
    x1, y1, x2, y2 = normalized
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def bbox_from_points(points: Sequence[Sequence[Any]]) -> BBox | None:
    normalized = normalize_points(points)
    if len(normalized) < 3:
        return None
    xs = [p[0] for p in normalized]
    ys = [p[1] for p in normalized]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x1 == x2 or y1 == y2:
        return None
    return x1, y1, x2, y2


def normalize_points(
    points: Sequence[Sequence[Any]] | None,
    image_shape: ImageShape | None = None,
    clamp: bool = True,
) -> list[Point]:
    if not points:
        return []

    normalized: list[Point] = []
    for point in points:
        if point is None or len(point) < 2:
            continue
        x = _coerce_int(point[0])
        y = _coerce_int(point[1])
        if x is None or y is None:
            continue
        if image_shape is not None and clamp:
            height, width = image_shape
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
        normalized.append((x, y))

    deduped: list[Point] = []
    for point in normalized:
        if not deduped or deduped[-1] != point:
            deduped.append(point)

    if len(deduped) > 2 and deduped[0] == deduped[-1]:
        deduped.pop()

    return deduped


def normalize_bbox(bbox: Sequence[Any] | None, image_shape: ImageShape | None = None) -> BBox | None:
    if bbox is None:
        return None
    return clamp_bbox(bbox, image_shape=image_shape)


def scale_points(points: Sequence[Sequence[Any]], scale_x: float, scale_y: float) -> list[Point]:
    scaled: list[Point] = []
    for point in points or []:
        if point is None or len(point) < 2:
            continue
        x = _coerce_int(point[0])
        y = _coerce_int(point[1])
        if x is None or y is None:
            continue
        scaled.append((_coerce_int(x * scale_x) or 0, _coerce_int(y * scale_y) or 0))
    return scaled


def scale_bbox(bbox: Sequence[Any] | None, scale_x: float, scale_y: float) -> BBox | None:
    if bbox is None:
        return None
    normalized = clamp_bbox(bbox)
    if normalized is None:
        return None
    x1, y1, x2, y2 = normalized
    scaled = (
        _coerce_int(x1 * scale_x),
        _coerce_int(y1 * scale_y),
        _coerce_int(x2 * scale_x),
        _coerce_int(y2 * scale_y),
    )
    if None in scaled:
        return None
    return clamp_bbox(scaled)  # type: ignore[arg-type]


def slot_center(slot: Mapping[str, Any]) -> Point | None:
    bbox = normalize_bbox(slot.get("bbox"))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    points = normalize_points(slot.get("points"))
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) // len(xs), sum(ys) // len(ys))


def slot_area(slot: Mapping[str, Any]) -> int:
    bbox = normalize_bbox(slot.get("bbox"))
    if bbox is None:
        bbox = bbox_from_points(slot.get("points", []))
    if bbox is None:
        return 0
    x1, y1, x2, y2 = bbox
    return abs((x2 - x1) * (y2 - y1))


def canonicalize_slot(
    slot: Mapping[str, Any],
    image_shape: ImageShape | None = None,
    *,
    default_id: int | str | None = None,
    clamp: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(slot, Mapping):
        return None

    points = normalize_points(slot.get("points"), image_shape=image_shape, clamp=clamp)
    bbox = normalize_bbox(slot.get("bbox"), image_shape=image_shape)

    if len(points) < 3 and bbox is not None:
        points = points_from_bbox(bbox)
    if bbox is None and len(points) >= 3:
        bbox = bbox_from_points(points)
    if bbox is None or len(points) < 3:
        return None

    center = slot_center({"bbox": bbox, "points": points})
    score_value = slot.get("score", slot.get("confidence", 0.0))
    score = float(score_value) if score_value is not None else 0.0

    canonical: dict[str, Any] = {
        "id": slot.get("id", default_id),
        "points": [[x, y] for x, y in points],
        "bbox": list(bbox),
        "center": list(center) if center is not None else None,
        "row": slot.get("row"),
        "score": score,
        "confidence": score,
    }

    if canonical["id"] is None and default_id is not None:
        canonical["id"] = default_id

    return canonical


def normalize_slots(
    slots: Sequence[Mapping[str, Any]] | None,
    image_shape: ImageShape | None = None,
    *,
    clamp: bool = True,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, slot in enumerate(slots or []):
        canonical = canonicalize_slot(slot, image_shape=image_shape, default_id=index, clamp=clamp)
        if canonical is not None:
            normalized.append(canonical)
    return normalized


def scale_slot(
    slot: Mapping[str, Any],
    scale_x: float,
    scale_y: float,
) -> dict[str, Any] | None:
    if not isinstance(slot, Mapping):
        return None

    points = slot.get("points")
    bbox = slot.get("bbox")

    scaled_points = None
    if points:
        scaled_points = scale_points(points, scale_x, scale_y)
    scaled_bbox = scale_bbox(bbox, scale_x, scale_y)

    if scaled_bbox is None and scaled_points:
        scaled_bbox = bbox_from_points(scaled_points)
    if scaled_points is None and scaled_bbox is not None:
        scaled_points = points_from_bbox(scaled_bbox)
    if scaled_bbox is None or not scaled_points:
        return None

    center = slot_center({"bbox": scaled_bbox, "points": scaled_points})
    return {
        "id": slot.get("id"),
        "points": [[x, y] for x, y in scaled_points],
        "bbox": list(scaled_bbox),
        "center": list(center) if center is not None else None,
        "row": slot.get("row"),
        "score": float(slot.get("score", slot.get("confidence", 0.0)) or 0.0),
        "confidence": float(slot.get("score", slot.get("confidence", 0.0)) or 0.0),
    }


def scale_slots(slots: Sequence[Mapping[str, Any]], scale_x: float, scale_y: float) -> list[dict[str, Any]]:
    scaled: list[dict[str, Any]] = []
    for slot in slots or []:
        normalized = scale_slot(slot, scale_x, scale_y)
        if normalized is not None:
            scaled.append(normalized)
    return scaled


def validate_slot(
    slot: Mapping[str, Any],
    image_shape: ImageShape | None = None,
    *,
    min_area: int = 1,
) -> bool:
    canonical = canonicalize_slot(slot, image_shape=image_shape)
    if canonical is None:
      return False
    return slot_area(canonical) >= int(min_area)
