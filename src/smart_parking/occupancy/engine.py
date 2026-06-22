"""Occupancy engine built on top of the existing SmartParkingV2 detector.

The engine accepts canonical slot polygons from any source, applies temporal
smoothing to per-slot occupancy, and returns counts plus per-slot statuses.
It is intentionally agnostic to whether slots came from a manual JSON file or
an auto-generated cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np


def _load_default_backend():
    # Lazy import keeps module import cheap and allows injection for tests.
    from smart_parking.detection.manual_detector import SmartParkingV2

    return SmartParkingV2()


def _as_int_point(point: Any) -> tuple[int, int]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"Invalid point: {point!r}")
    return int(round(point[0])), int(round(point[1]))


def _bbox_from_points(points: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _rect_points_from_bbox(bbox: Sequence[int]) -> list[tuple[int, int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox!r}")
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _clip_points(
    points: Sequence[tuple[int, int]],
    frame_shape: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    if frame_shape is None:
        return [(int(x), int(y)) for x, y in points]

    height, width = frame_shape
    clipped = []
    for x, y in points:
        clipped.append(
            (
                max(0, min(int(x), width - 1)),
                max(0, min(int(y), height - 1)),
            )
        )
    return clipped


def _scale_points(
    points: Sequence[tuple[int, int]],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    src_h, src_w = source_shape
    dst_h, dst_w = target_shape
    if src_h <= 0 or src_w <= 0:
        return [(int(x), int(y)) for x, y in points]

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    return [
        (int(round(x * scale_x)), int(round(y * scale_y)))
        for x, y in points
    ]


def _extract_source_shape(slots: Any) -> tuple[int, int] | None:
    if isinstance(slots, Mapping):
        shape = slots.get("image_shape") or slots.get("frame_shape")
        if shape and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    return None


def _extract_slot_items(slots: Any) -> list[Mapping[str, Any]]:
    if isinstance(slots, Mapping):
        raw_slots = slots.get("slots", slots)
        if isinstance(raw_slots, Mapping):
            raw_slots = list(raw_slots.values())
        return [slot for slot in raw_slots if isinstance(slot, Mapping)]

    if isinstance(slots, Iterable) and not isinstance(slots, (str, bytes)):
        return [slot for slot in slots if isinstance(slot, Mapping)]

    raise TypeError("slots must be a mapping with 'slots' or an iterable of slot mappings")


@dataclass(frozen=True)
class NormalizedSlot:
    """Canonical slot polygon used by the occupancy engine."""

    id: str
    points: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    row: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class SlotOccupancyStatus:
    """Per-slot output returned by the occupancy engine."""

    id: str
    occupied: bool
    unsure: bool
    status: str
    confidence: float
    overlap_threshold: float | None
    occupancy_threshold: float | None
    label: str
    points: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    raw_status: str
    smoothed_state: int


@dataclass(frozen=True)
class OccupancyResult:
    """Result bundle returned by :class:`OccupancyEngine`."""

    total_slots: int
    occupied: int
    available: int
    unsure: int
    slots: tuple[SlotOccupancyStatus, ...]
    vehicle_boxes: tuple[tuple[int, int, int, int], ...] = tuple()
    person_boxes: tuple[tuple[int, int, int, int], ...] = tuple()
    annotated_frame: np.ndarray | None = field(default=None, repr=False)


@dataclass
class _SlotState:
    state: int | None = None
    streak: int = 0


def normalize_slots(
    slots: Any,
    frame_shape: tuple[int, int] | None = None,
) -> list[NormalizedSlot]:
    """Normalize manual JSON or auto-cache slot payloads into canonical slots."""

    source_shape = _extract_source_shape(slots)
    slot_items = _extract_slot_items(slots)

    normalized: list[NormalizedSlot] = []
    for index, slot in enumerate(slot_items):
        slot_id = slot.get("id", index)
        slot_id_str = str(slot_id)

        points = slot.get("points")
        if points is None and "bbox" in slot:
            points = _rect_points_from_bbox(slot["bbox"])

        if points is None:
            continue

        canonical_points = [_as_int_point(point) for point in points]
        if source_shape and frame_shape and source_shape != frame_shape:
            canonical_points = _scale_points(canonical_points, source_shape, frame_shape)

        canonical_points = _clip_points(canonical_points, frame_shape)
        if len(canonical_points) < 3:
            continue

        bbox = slot.get("bbox")
        if bbox is not None:
            if source_shape and frame_shape and source_shape != frame_shape:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                scale_x = frame_shape[1] / source_shape[1] if source_shape[1] else 1.0
                scale_y = frame_shape[0] / source_shape[0] if source_shape[0] else 1.0
                bbox = (
                    int(round(x1 * scale_x)),
                    int(round(y1 * scale_y)),
                    int(round(x2 * scale_x)),
                    int(round(y2 * scale_y)),
                )
            else:
                bbox = tuple(int(round(v)) for v in bbox)
        else:
            bbox = _bbox_from_points(canonical_points)

        if frame_shape is not None:
            bbox = (
                max(0, min(bbox[0], frame_shape[1] - 1)),
                max(0, min(bbox[1], frame_shape[0] - 1)),
                max(0, min(bbox[2], frame_shape[1] - 1)),
                max(0, min(bbox[3], frame_shape[0] - 1)),
            )

        metadata = {
            key: value
            for key, value in slot.items()
            if key not in {"id", "points", "bbox"}
        }
        normalized.append(
            NormalizedSlot(
                id=slot_id_str,
                points=tuple(canonical_points),
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                row=int(slot["row"]) if "row" in slot and slot["row"] is not None else None,
                score=float(slot["score"]) if "score" in slot and slot["score"] is not None else None,
                metadata=metadata,
            )
        )

    return normalized


class OccupancyEngine:
    """Temporal occupancy engine for canonical parking-slot polygons."""

    def __init__(
        self,
        backend: Any | None = None,
        backend_factory: Callable[[], Any] | None = None,
        *,
        conf_threshold: float = 0.2,
        overlap_threshold: float = 0.2,
        imgsz: int = 1280,
        pad_ratio: float = 0.1,
        unsure_threshold: float = 0.8,
        smooth_frames: int = 3,
        dynamic_occupancy_threshold: bool = False,
        near_occupancy_threshold: float = 0.55,
        far_occupancy_threshold: float = 0.30,
        dynamic_threshold_axis: str = "y",
        show_thresholds: bool = False,
    ) -> None:
        self._backend = backend
        self._backend_factory = backend_factory or _load_default_backend
        self.conf_threshold = conf_threshold
        self.overlap_threshold = overlap_threshold
        self.imgsz = imgsz
        self.pad_ratio = pad_ratio
        self.unsure_threshold = unsure_threshold
        self.smooth_frames = max(1, int(smooth_frames))
        self.dynamic_occupancy_threshold = bool(dynamic_occupancy_threshold)
        self.near_occupancy_threshold = float(near_occupancy_threshold)
        self.far_occupancy_threshold = float(far_occupancy_threshold)
        self.dynamic_threshold_axis = dynamic_threshold_axis
        self.show_thresholds = bool(show_thresholds)
        self._slot_state: dict[str, _SlotState] = {}

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def reset(self) -> None:
        """Clear temporal occupancy history."""

        self._slot_state.clear()

    def process_frame(
        self,
        frame: np.ndarray,
        slots: Any,
        *,
        annotate: bool = True,
    ) -> OccupancyResult:
        if frame is None:
            raise ValueError("frame is None")

        frame_shape = frame.shape[:2]
        normalized_slots = normalize_slots(slots, frame_shape=frame_shape)
        if not normalized_slots:
            empty = frame.copy() if annotate else None
            return OccupancyResult(
                total_slots=0,
                occupied=0,
                available=0,
                unsure=0,
                slots=tuple(),
                vehicle_boxes=tuple(),
                person_boxes=tuple(),
                annotated_frame=empty,
            )

        backend_slots = [{"points": [list(point) for point in slot.points]} for slot in normalized_slots]
        _, raw_statuses, _, _, _, vehicle_boxes, person_boxes = self.backend.detect_with_slots_frame(
            frame,
            backend_slots,
            conf_threshold=self.conf_threshold,
            overlap_threshold=self.overlap_threshold,
            imgsz=self.imgsz,
            pad_ratio=self.pad_ratio,
            unsure_threshold=self.unsure_threshold,
            dynamic_occupancy_threshold=self.dynamic_occupancy_threshold,
            near_occupancy_threshold=self.near_occupancy_threshold,
            far_occupancy_threshold=self.far_occupancy_threshold,
            dynamic_threshold_axis=self.dynamic_threshold_axis,
        )

        statuses = self._smooth_statuses(normalized_slots, raw_statuses)
        annotated_frame = self._draw_statuses(frame, statuses) if annotate else None

        occupied = sum(1 for status in statuses if status.status == "occupied")
        unsure = sum(1 for status in statuses if status.unsure)
        available = len(statuses) - occupied - unsure

        return OccupancyResult(
            total_slots=len(statuses),
            occupied=occupied,
            available=available,
            unsure=unsure,
            slots=tuple(statuses),
            vehicle_boxes=tuple(
                tuple(int(value) for value in box)
                for box in vehicle_boxes
                if isinstance(box, (list, tuple)) and len(box) == 4
            ),
            person_boxes=tuple(
                tuple(int(value) for value in box)
                for box in person_boxes
                if isinstance(box, (list, tuple)) and len(box) == 4
            ),
            annotated_frame=annotated_frame,
        )

    def annotate_statuses(
        self,
        frame: np.ndarray,
        statuses: Sequence[SlotOccupancyStatus],
    ) -> np.ndarray:
        """Draw the latest known slot statuses on a fresh frame."""

        return self._draw_statuses(frame, statuses)

    def _smooth_statuses(
        self,
        slots: Sequence[NormalizedSlot],
        raw_statuses: Sequence[Mapping[str, Any]],
    ) -> list[SlotOccupancyStatus]:
        smoothed: list[SlotOccupancyStatus] = []

        for index, slot in enumerate(slots):
            raw = raw_statuses[index] if index < len(raw_statuses) else {}
            raw_occupied = bool(raw.get("occupied", False))
            raw_unsure = bool(raw.get("unsure", False))
            confidence = float(raw.get("confidence", 0.0) or 0.0)
            overlap_threshold = raw.get("overlap_threshold")
            overlap_threshold = float(overlap_threshold) if overlap_threshold is not None else None
            occupancy_threshold = raw.get("occupancy_threshold")
            occupancy_threshold = float(occupancy_threshold) if occupancy_threshold is not None else None

            state = self._slot_state.setdefault(slot.id, _SlotState())
            proposed_state = 1 if raw_occupied else 0

            if state.state is None:
                state.state = proposed_state
                state.streak = 0
            elif raw_unsure:
                state.streak = 0
            elif proposed_state == state.state:
                state.streak = 0
            else:
                state.streak += 1
                if state.streak >= self.smooth_frames:
                    state.state = proposed_state
                    state.streak = 0

            current_state = state.state if state.state is not None else proposed_state
            if raw_unsure:
                status = "unsure"
                label = f"#{index + 1} UNSURE"
            elif current_state == 1:
                status = "occupied"
                label = f"#{index + 1} OCCUPIED"
            else:
                status = "available"
                label = f"#{index + 1} AVAILABLE"

            smoothed.append(
                SlotOccupancyStatus(
                    id=slot.id,
                    occupied=status == "occupied",
                    unsure=raw_unsure,
                    status=status,
                    confidence=confidence,
                    overlap_threshold=overlap_threshold,
                    occupancy_threshold=occupancy_threshold,
                    label=label,
                    points=slot.points,
                    bbox=slot.bbox,
                    raw_status="unsure" if raw_unsure else ("occupied" if raw_occupied else "available"),
                    smoothed_state=int(current_state),
                )
            )

        return smoothed

    def _draw_statuses(
        self,
        frame: np.ndarray,
        statuses: Sequence[SlotOccupancyStatus],
    ) -> np.ndarray:
        output = frame.copy()
        for index, status in enumerate(statuses):
            points = np.array(status.points, dtype=np.int32)
            if status.unsure:
                color = (0, 255, 255)
            elif status.occupied:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.polylines(output, [points], True, color, 3)
            overlay = output.copy()
            cv2.fillPoly(overlay, [points], color)
            output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0)

            centroid = points.mean(axis=0).astype(int)
            cv2.putText(
                output,
                status.label,
                (centroid[0] - 50, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            if status.confidence > 0:
                conf_text = f"{status.confidence * 100:.1f}%"
                if self.show_thresholds and status.occupancy_threshold is not None:
                    conf_text = f"{conf_text}/T{status.occupancy_threshold * 100:.0f}%"
                cv2.putText(
                    output,
                    conf_text,
                    (centroid[0] - 20, centroid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        return output
