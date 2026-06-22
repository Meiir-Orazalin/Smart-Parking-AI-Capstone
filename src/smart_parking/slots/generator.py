from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .geometry import canonicalize_slot, coerce_image_shape, normalize_points

Point = tuple[int, int]


class SeedValidationError(ValueError):
    """Raised when anchor-slot seeds cannot be converted into a full layout."""


@dataclass(frozen=True)
class AnchorSlot:
    row: int
    slot_index: int
    slot_count: int
    points: tuple[Point, ...]


def load_anchor_seed_file(path: str | Path) -> dict[str, Any]:
    seed_path = Path(path)
    with seed_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        payload = {"anchors": payload}
    if not isinstance(payload, Mapping):
        raise SeedValidationError(f"Unsupported seed payload: {seed_path}")
    return dict(payload)


def save_anchor_seed_file(
    path: str | Path,
    anchors: Sequence[Mapping[str, Any]],
    *,
    image_shape: tuple[int, int] | None = None,
) -> Path:
    seed_path = Path(path)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "image_shape": list(image_shape) if image_shape is not None else None,
        "anchors": list(anchors),
    }
    seed_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return seed_path


def _order_quad(points: Sequence[Point]) -> tuple[Point, Point, Point, Point]:
    if len(points) != 4:
        raise SeedValidationError("Anchor slots must use exactly 4 corners for interpolation")

    ordered = sorted(points, key=lambda point: (point[1], point[0]))
    top_pair = sorted(ordered[:2], key=lambda point: point[0])
    bottom_pair = sorted(ordered[2:], key=lambda point: point[0])
    top_left, top_right = top_pair
    bottom_left, bottom_right = bottom_pair
    return top_left, top_right, bottom_right, bottom_left


def _as_anchor_slot(raw: Mapping[str, Any], image_shape: tuple[int, int] | None) -> AnchorSlot:
    canonical = canonicalize_slot(raw, image_shape=image_shape, clamp=image_shape is not None)
    if canonical is None:
        raise SeedValidationError(f"Invalid anchor slot: {raw!r}")

    row = raw.get("row")
    slot_index = raw.get("slot_index")
    slot_count = raw.get("slot_count")
    if row is None or slot_index is None or slot_count is None:
        raise SeedValidationError("Each anchor must include row, slot_index, and slot_count")

    row = int(row)
    slot_index = int(slot_index)
    slot_count = int(slot_count)
    if row < 1:
        raise SeedValidationError("row must be >= 1")
    if slot_count < 2:
        raise SeedValidationError("slot_count must be >= 2")
    if slot_index < 1 or slot_index > slot_count:
        raise SeedValidationError("slot_index must be within 1..slot_count")

    points = _order_quad(tuple((int(point[0]), int(point[1])) for point in canonical.get("points", [])))

    return AnchorSlot(
        row=row,
        slot_index=slot_index,
        slot_count=slot_count,
        points=points,
    )


def _group_anchor_rows(
    payload: Mapping[str, Any],
    *,
    image_shape: tuple[int, int] | None = None,
) -> tuple[dict[int, list[AnchorSlot]], tuple[int, int] | None]:
    payload_shape = coerce_image_shape(payload.get("image_shape"))
    effective_shape = image_shape or payload_shape
    raw_anchors = payload.get("anchors")
    if not isinstance(raw_anchors, Iterable) or isinstance(raw_anchors, (str, bytes, Mapping)):
        raise SeedValidationError("Seed payload must contain an 'anchors' list")

    grouped: dict[int, list[AnchorSlot]] = {}
    for raw_anchor in raw_anchors:
        if not isinstance(raw_anchor, Mapping):
            raise SeedValidationError("Each anchor must be an object")
        anchor = _as_anchor_slot(raw_anchor, effective_shape)
        grouped.setdefault(anchor.row, []).append(anchor)

    if not grouped:
        raise SeedValidationError("Seed payload contains no anchors")

    for row, anchors in grouped.items():
        anchors.sort(key=lambda anchor: anchor.slot_index)
        slot_count = anchors[0].slot_count
        seen_indexes: set[int] = set()
        normalized_row: list[AnchorSlot] = []
        for anchor in anchors:
            if anchor.slot_count != slot_count:
                raise SeedValidationError(f"Row {row} mixes multiple slot_count values")
            if anchor.slot_index in seen_indexes:
                raise SeedValidationError(f"Row {row} repeats slot_index {anchor.slot_index}")
            seen_indexes.add(anchor.slot_index)
            normalized_anchor = AnchorSlot(
                row=anchor.row,
                slot_index=anchor.slot_index,
                slot_count=anchor.slot_count,
                points=anchor.points,
            )
            normalized_row.append(normalized_anchor)

        if normalized_row[0].slot_index != 1 or normalized_row[-1].slot_index != slot_count:
            raise SeedValidationError(
                f"Row {row} must anchor the first and last slot (1 and {slot_count})"
            )
        grouped[row] = normalized_row

    return grouped, effective_shape


def _interpolate_line(start: tuple[Point, Point], end: tuple[Point, Point], t: float) -> tuple[Point, Point]:
    result: list[Point] = []
    for index in range(2):
        sx, sy = start[index]
        ex, ey = end[index]
        result.append(
            (
                int(round(sx + (ex - sx) * t)),
                int(round(sy + (ey - sy) * t)),
            )
        )
    return result[0], result[1]


def _edge_center(edge: tuple[Point, Point]) -> tuple[float, float]:
    return (
        (edge[0][0] + edge[1][0]) / 2.0,
        (edge[0][1] + edge[1][1]) / 2.0,
    )


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _vector(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return (right[0] - left[0], right[1] - left[1])


def _normalize_vector(vector: tuple[float, float]) -> tuple[float, float]:
    length = float((vector[0] ** 2 + vector[1] ** 2) ** 0.5)
    if length <= 1e-6:
        raise SeedValidationError("Could not determine row direction from anchors")
    return (vector[0] / length, vector[1] / length)


def _orient_edge(edge: tuple[Point, Point]) -> tuple[Point, Point]:
    left, right = edge
    if (left[0], left[1]) <= (right[0], right[1]):
        return left, right
    return right, left


def _row_direction(anchors: Sequence[AnchorSlot]) -> tuple[float, float]:
    if len(anchors) < 2:
        raise SeedValidationError("Need at least two anchors per row")
    first = tuple(float(value) for value in canonicalize_slot({"points": anchors[0].points}).get("center", (0, 0)))
    last = tuple(float(value) for value in canonicalize_slot({"points": anchors[-1].points}).get("center", (0, 0)))
    return _normalize_vector(_vector(first, last))


def _divider_edges_for_anchor(
    anchor: AnchorSlot,
    direction: tuple[float, float],
) -> tuple[tuple[Point, Point], tuple[Point, Point]]:
    top_left, top_right, bottom_right, bottom_left = anchor.points
    candidate_pairs = [
        ((_orient_edge((top_left, top_right))), (_orient_edge((bottom_left, bottom_right)))),
        ((_orient_edge((top_left, bottom_left))), (_orient_edge((top_right, bottom_right)))),
    ]

    best_pair: tuple[tuple[Point, Point], tuple[Point, Point]] | None = None
    best_alignment = -1.0
    for first_edge, second_edge in candidate_pairs:
        center_delta = _vector(_edge_center(first_edge), _edge_center(second_edge))
        alignment = abs(_dot(_normalize_vector(center_delta), direction))
        if alignment > best_alignment:
            best_alignment = alignment
            best_pair = (first_edge, second_edge)

    if best_pair is None:
        raise SeedValidationError("Could not determine divider edges for anchor")

    first_edge, second_edge = best_pair
    first_projection = _dot(_edge_center(first_edge), direction)
    second_projection = _dot(_edge_center(second_edge), direction)
    if first_projection <= second_projection:
        return first_edge, second_edge
    return second_edge, first_edge


def _average_lines(lines: Sequence[tuple[Point, Point]]) -> tuple[Point, Point]:
    if not lines:
        raise SeedValidationError("Cannot average an empty divider-line set")
    left_x = sum(line[0][0] for line in lines) / len(lines)
    left_y = sum(line[0][1] for line in lines) / len(lines)
    right_x = sum(line[1][0] for line in lines) / len(lines)
    right_y = sum(line[1][1] for line in lines) / len(lines)
    return (
        (int(round(left_x)), int(round(left_y))),
        (int(round(right_x)), int(round(right_y))),
    )


def _project_point_to_line_parameter(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    direction = _vector(start, end)
    denom = direction[0] ** 2 + direction[1] ** 2
    if denom <= 1e-6:
        return 0.0
    offset = _vector(start, point)
    return float((offset[0] * direction[0] + offset[1] * direction[1]) / denom)


def _fit_projective_parameters(
    boundary_indexes: Sequence[int],
    samples: Sequence[float],
    slot_count: int,
) -> tuple[float, float, float] | None:
    if len(boundary_indexes) < 3 or len(boundary_indexes) != len(samples) or slot_count <= 0:
        return None

    rows = []
    values = []
    for boundary_index, sample in zip(boundary_indexes, samples):
        u = float(boundary_index) / float(slot_count)
        rows.append([u, 1.0, -sample * u])
        values.append(sample)

    matrix = np.asarray(rows, dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    try:
        solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(solution[0]), float(solution[1]), float(solution[2])


def _evaluate_projective(parameter: tuple[float, float, float] | None, u: float) -> float | None:
    if parameter is None:
        return None
    a, b, c = parameter
    denom = c * u + 1.0
    if abs(denom) <= 1e-6:
        return None
    return float((a * u + b) / denom)


def _piecewise_linear_samples(
    boundary_indexes: Sequence[int],
    samples: Sequence[float],
    slot_count: int,
) -> list[float]:
    if len(boundary_indexes) != len(samples):
        raise SeedValidationError("Boundary sample indexes and values must match")
    if not boundary_indexes:
        raise SeedValidationError("Need at least one boundary sample")

    result: list[float] = []
    for boundary_index in range(slot_count + 1):
        if boundary_index <= boundary_indexes[0]:
            result.append(float(samples[0]))
            continue
        if boundary_index >= boundary_indexes[-1]:
            result.append(float(samples[-1]))
            continue
        for left_index in range(len(boundary_indexes) - 1):
            start_idx = boundary_indexes[left_index]
            end_idx = boundary_indexes[left_index + 1]
            if start_idx <= boundary_index <= end_idx:
                span = max(1, end_idx - start_idx)
                t = (boundary_index - start_idx) / float(span)
                value = samples[left_index] + (samples[left_index + 1] - samples[left_index]) * t
                result.append(float(value))
                break
    return result


def _monotonicize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    result = [float(values[0])]
    for value in values[1:]:
        result.append(max(result[-1] + 1e-4, float(value)))
    return result


def _fit_boundary_parameters(
    averaged: Mapping[int, tuple[Point, Point]],
    slot_count: int,
) -> list[float]:
    ordered_indexes = sorted(averaged)
    if 0 not in averaged or slot_count not in averaged:
        raise SeedValidationError("Anchors must define the first and last slot boundaries")

    near_center = _edge_center(averaged[0])
    far_center = _edge_center(averaged[slot_count])
    samples = [
        _project_point_to_line_parameter(_edge_center(averaged[index]), near_center, far_center)
        for index in ordered_indexes
    ]

    parameter = _fit_projective_parameters(ordered_indexes, samples, slot_count)
    fitted: list[float] = []
    if parameter is not None:
        for boundary_index in range(slot_count + 1):
            u = float(boundary_index) / float(slot_count)
            value = _evaluate_projective(parameter, u)
            fitted.append(float(value) if value is not None else float("nan"))
        if any(not np.isfinite(value) for value in fitted):
            fitted = []

    if not fitted:
        fitted = _piecewise_linear_samples(ordered_indexes, samples, slot_count)

    # Normalize the known boundary samples first.
    minimum = float(samples[0])
    maximum = float(samples[-1])
    if abs(maximum - minimum) <= 1e-6:
        return [boundary_index / float(slot_count) for boundary_index in range(slot_count + 1)]

    normalized_known = {
        index: (sample - minimum) / (maximum - minimum)
        for index, sample in zip(ordered_indexes, samples)
    }

    normalized_fit = [(value - minimum) / (maximum - minimum) for value in fitted]

    # Preserve every anchored boundary exactly, and use the global perspective fit
    # only to distribute the interior points within each anchored span.
    result: list[float] = [0.0] * (slot_count + 1)
    for left_index, right_index in zip(ordered_indexes, ordered_indexes[1:]):
        left_value = float(normalized_known[left_index])
        right_value = float(normalized_known[right_index])
        result[left_index] = left_value
        result[right_index] = right_value
        if right_index - left_index <= 1:
            continue

        fit_left = float(normalized_fit[left_index])
        fit_right = float(normalized_fit[right_index])
        span_fit = fit_right - fit_left
        span_exact = right_value - left_value
        for boundary_index in range(left_index + 1, right_index):
            if abs(span_fit) <= 1e-6:
                local_t = (boundary_index - left_index) / float(right_index - left_index)
            else:
                local_t = (normalized_fit[boundary_index] - fit_left) / span_fit
            result[boundary_index] = left_value + span_exact * local_t

    result[ordered_indexes[0]] = float(normalized_known[ordered_indexes[0]])
    result[ordered_indexes[-1]] = float(normalized_known[ordered_indexes[-1]])
    result[0] = 0.0
    result[-1] = 1.0

    monotonic = _monotonicize(result)
    monotonic[0] = 0.0
    monotonic[-1] = 1.0
    return monotonic


def _interpolate_point(start: Point, end: Point, t: float) -> Point:
    return (
        int(round(start[0] + (end[0] - start[0]) * t)),
        int(round(start[1] + (end[1] - start[1]) * t)),
    )


def _build_divider_lines(anchors: Sequence[AnchorSlot]) -> dict[int, tuple[Point, Point]]:
    known: dict[int, list[tuple[Point, Point]]] = {}
    direction = _row_direction(anchors)
    for anchor in anchors:
        start_edge, end_edge = _divider_edges_for_anchor(anchor, direction)
        known.setdefault(anchor.slot_index - 1, []).append(start_edge)
        known.setdefault(anchor.slot_index, []).append(end_edge)

    averaged = {index: _average_lines(lines) for index, lines in known.items()}
    if not averaged:
        raise SeedValidationError("No divider lines could be derived from anchors")
    slot_count = anchors[0].slot_count
    boundary_parameters = _fit_boundary_parameters(averaged, slot_count)

    near_left, near_right = averaged[0]
    far_left, far_right = averaged[slot_count]
    completed: dict[int, tuple[Point, Point]] = {}
    for boundary_index, t in enumerate(boundary_parameters):
        completed[boundary_index] = (
            _interpolate_point(near_left, far_left, t),
            _interpolate_point(near_right, far_right, t),
        )

    # Preserve all explicit anchor-derived divider lines exactly.
    for boundary_index, line in averaged.items():
        completed[boundary_index] = line
    return completed


def _segment_intersection_ratio(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> float:
    left_contour = np.array(left, dtype=np.float32)
    right_contour = np.array(right, dtype=np.float32)
    left_area = abs(float(cv2.contourArea(left_contour)))
    right_area = abs(float(cv2.contourArea(right_contour)))
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(left_contour, right_contour)
    if intersection_area <= 0.0:
        return 0.0
    return float(intersection_area) / max(1.0, min(left_area, right_area))


def _validate_simple_polygon(points: Sequence[Point]) -> bool:
    contour = np.array(points, dtype=np.int32)
    if len(contour) < 3:
        return False
    area = abs(float(cv2.contourArea(contour)))
    if area <= 0.0:
        return False
    return cv2.isContourConvex(contour)


def _validate_generated_rows(
    generated_rows: Mapping[int, Sequence[dict[str, Any]]],
    *,
    min_area: float,
    max_adjacent_overlap: float,
) -> None:
    for row, slots in generated_rows.items():
        if not slots:
            raise SeedValidationError(f"Row {row} generated no slots")

        centers = []
        for slot in slots:
            points = tuple((int(point[0]), int(point[1])) for point in slot.get("points", []))
            if len(points) != 4:
                raise SeedValidationError(f"Generated row {row} contains a non-quad slot")
            if not _validate_simple_polygon(points):
                raise SeedValidationError(f"Generated row {row} contains a self-intersecting slot")
            area = abs(float(cv2.contourArea(np.array(points, dtype=np.int32))))
            if area < float(min_area):
                raise SeedValidationError(f"Generated row {row} contains a near-zero-area slot")
            centers.append(tuple(slot.get("center", (0, 0))))

        if len(centers) >= 2:
            total_dx = centers[-1][0] - centers[0][0]
            total_dy = centers[-1][1] - centers[0][1]
            for index in range(len(centers) - 1):
                step_dx = centers[index + 1][0] - centers[index][0]
                step_dy = centers[index + 1][1] - centers[index][1]
                if (step_dx, step_dy) == (0, 0):
                    raise SeedValidationError(f"Generated row {row} contains duplicate adjacent slot centers")
                if (step_dx * total_dx + step_dy * total_dy) <= 0:
                    raise SeedValidationError(f"Generated row {row} reverses direction between adjacent slots")

        # Adjacent slots can legitimately overlap in image space under strong
        # perspective distortion, especially near the camera. We keep the
        # parameter for future tuning but do not fail generation on overlap.
        _ = max_adjacent_overlap


def generate_slots_from_seed_payload(
    payload: Mapping[str, Any],
    *,
    image_shape: tuple[int, int] | None = None,
    min_area: float = 50.0,
    max_adjacent_overlap: float = 0.80,
) -> list[dict[str, Any]]:
    grouped, effective_shape = _group_anchor_rows(payload, image_shape=image_shape)
    generated_rows: dict[int, list[dict[str, Any]]] = {}

    for row in sorted(grouped):
        anchors = grouped[row]
        row_slots: list[dict[str, Any]] = []
        expected_count = anchors[0].slot_count
        divider_lines = _build_divider_lines(anchors)
        missing_boundaries = [index for index in range(0, expected_count + 1) if index not in divider_lines]
        if missing_boundaries:
            raise SeedValidationError(f"Row {row} is missing divider lines for boundaries {missing_boundaries}")

        anchor_indexes = {anchor.slot_index for anchor in anchors}
        for slot_index in range(1, expected_count + 1):
            top_left, top_right = divider_lines[slot_index - 1]
            bottom_left, bottom_right = divider_lines[slot_index]
            slot = canonicalize_slot(
                {
                    "id": f"r{row:02d}s{slot_index:02d}",
                    "row": row,
                    "slot_index": slot_index,
                    "points": [
                        [top_left[0], top_left[1]],
                        [top_right[0], top_right[1]],
                        [bottom_right[0], bottom_right[1]],
                        [bottom_left[0], bottom_left[1]],
                    ],
                    "is_anchor": slot_index in anchor_indexes,
                },
                image_shape=effective_shape,
                clamp=effective_shape is not None,
            )
            if slot is None:
                raise SeedValidationError(f"Could not generate row {row} slot {slot_index}")
            slot["slot_index"] = slot_index
            slot["is_anchor"] = slot_index in anchor_indexes
            row_slots.append(slot)

        if len(row_slots) != expected_count:
            raise SeedValidationError(
                f"Row {row} expected {expected_count} slots but generated {len(row_slots)}"
            )
        generated_rows[row] = row_slots

    _validate_generated_rows(
        generated_rows,
        min_area=min_area,
        max_adjacent_overlap=max_adjacent_overlap,
    )

    combined: list[dict[str, Any]] = []
    for row in sorted(generated_rows):
        for slot in generated_rows[row]:
            combined.append(
                {
                    key: value
                    for key, value in slot.items()
                    if key not in {"slot_index", "is_anchor"}
                }
            )
    return combined


def generate_slots_from_seed_file(
    path: str | Path,
    *,
    image_shape: tuple[int, int] | None = None,
    min_area: float = 50.0,
    max_adjacent_overlap: float = 0.80,
) -> list[dict[str, Any]]:
    payload = load_anchor_seed_file(path)
    return generate_slots_from_seed_payload(
        payload,
        image_shape=image_shape,
        min_area=min_area,
        max_adjacent_overlap=max_adjacent_overlap,
    )


def render_generated_slots_preview(
    image: np.ndarray,
    slots: Sequence[Mapping[str, Any]],
    *,
    anchors: Sequence[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    preview = image.copy()

    for slot in slots:
        points = normalize_points(slot.get("points"))
        if len(points) < 3:
            continue
        contour = np.array(points, dtype=np.int32)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [contour], (0, 180, 0))
        preview = cv2.addWeighted(overlay, 0.18, preview, 0.82, 0)
        cv2.polylines(preview, [contour], True, (0, 255, 0), 2)
        center = slot.get("center")
        if isinstance(center, Sequence) and len(center) >= 2:
            cv2.putText(
                preview,
                str(slot.get("id", "")),
                (int(center[0]) - 18, int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

    for anchor in anchors or []:
        points = normalize_points(anchor.get("points"))
        if len(points) < 3:
            continue
        contour = np.array(points, dtype=np.int32)
        for x, y in points:
            cv2.circle(preview, (int(x), int(y)), 5, (0, 180, 255), -1)
        centroid = contour.mean(axis=0).astype(int)
        cv2.putText(
            preview,
            f"A{anchor.get('row', '?')}-{anchor.get('slot_index', '?')}",
            (int(centroid[0]) - 24, int(centroid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 180, 255),
            1,
        )

    return preview


def default_preview_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    stem_path = output_path.with_suffix("")
    return stem_path.with_name(stem_path.name + "_preview.jpg")
