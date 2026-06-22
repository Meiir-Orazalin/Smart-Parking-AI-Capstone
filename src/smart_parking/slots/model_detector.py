from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .cache import canonicalize_slots
from .geometry import bbox_from_points


class YOLOSlotPolygonDetector:
    """Model-backed parking-slot detector using a segmentation-capable YOLO model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        conf_threshold: float = 0.25,
        min_area_ratio: float = 0.0005,
        max_area_ratio: float = 0.2,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Slot model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def detect_slots(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        if result.masks is None or result.boxes is None:
            return []

        frame_area = frame.shape[0] * frame.shape[1]
        slots: list[dict[str, Any]] = []
        polygons = result.masks.xy
        confidences = [float(box.conf[0]) for box in result.boxes]

        for index, polygon in enumerate(polygons):
            if polygon is None or len(polygon) < 3:
                continue
            contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            points = [[int(round(x)), int(round(y))] for x, y in simplified]
            area = abs(cv2.contourArea(np.array(points, dtype=np.int32)))
            if area <= 0:
                continue
            area_ratio = area / max(1, frame_area)
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            slots.append(
                {
                    "id": str(index),
                    "points": points,
                    "bbox": bbox_from_points(points),
                    "score": confidences[index] if index < len(confidences) else None,
                }
            )

        return canonicalize_slots(slots, image_shape=frame.shape[:2])
