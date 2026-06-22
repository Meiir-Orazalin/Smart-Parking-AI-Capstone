from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def clipped(self, frame_shape: tuple[int, int]) -> "Region | None":
        height, width = frame_shape
        clipped = Region(
            x1=max(0, min(self.x1, width)),
            y1=max(0, min(self.y1, height)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
        )
        if clipped.width <= 0 or clipped.height <= 0:
            return None
        return clipped


def _cascade_path(filename: str) -> str:
    return str(Path(cv2.data.haarcascades) / filename)


def _odd_kernel_size(length: int, *, minimum: int = 15, divisor: int = 3) -> int:
    if length <= 1:
        return 1
    target = max(minimum, length // max(1, divisor))
    if target % 2 == 0:
        target += 1
    max_allowed = length if length % 2 == 1 else max(1, length - 1)
    return max(1, min(target, max_allowed))


def _regions_overlap(left: Region, right: Region) -> bool:
    return not (
        left.x2 < right.x1
        or right.x2 < left.x1
        or left.y2 < right.y1
        or right.y2 < left.y1
    )


def _merge_pair(left: Region, right: Region) -> Region:
    return Region(
        x1=min(left.x1, right.x1),
        y1=min(left.y1, right.y1),
        x2=max(left.x2, right.x2),
        y2=max(left.y2, right.y2),
    )


def _merge_regions(regions: Iterable[Region]) -> list[Region]:
    pending = [region for region in regions if region.width > 0 and region.height > 0]
    merged: list[Region] = []
    while pending:
        current = pending.pop(0)
        index = 0
        while index < len(pending):
            if _regions_overlap(current, pending[index]):
                current = _merge_pair(current, pending.pop(index))
                index = 0
                continue
            index += 1
        merged.append(current)
    return merged


class OpenCVHaarAnonymizer:
    """Stateful face and license-plate anonymizer using bundled OpenCV cascades."""

    backend_name = "opencv_haar"

    def __init__(
        self,
        *,
        refresh_frames: int = 3,
        max_stale_cycles: int = 2,
        expand_ratio: float = 0.10,
        face_cascade_path: str | None = None,
        plate_cascade_path: str | None = None,
        face_scale_factor: float = 1.1,
        face_min_neighbors: int = 5,
        face_min_size: tuple[int, int] = (20, 20),
        plate_scale_factor: float = 1.1,
        plate_min_neighbors: int = 3,
        plate_min_size: tuple[int, int] = (20, 10),
    ) -> None:
        self.refresh_frames = max(1, int(refresh_frames))
        self.max_stale_cycles = max(0, int(max_stale_cycles))
        self.expand_ratio = max(0.0, float(expand_ratio))
        self.face_scale_factor = float(face_scale_factor)
        self.face_min_neighbors = int(face_min_neighbors)
        self.face_min_size = tuple(int(v) for v in face_min_size)
        self.plate_scale_factor = float(plate_scale_factor)
        self.plate_min_neighbors = int(plate_min_neighbors)
        self.plate_min_size = tuple(int(v) for v in plate_min_size)

        face_path = face_cascade_path or _cascade_path("haarcascade_frontalface_default.xml")
        plate_path = plate_cascade_path or _cascade_path("haarcascade_russian_plate_number.xml")
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.plate_cascade = cv2.CascadeClassifier(plate_path)

        failed = []
        if self.face_cascade.empty():
            failed.append(face_path)
        if self.plate_cascade.empty():
            failed.append(plate_path)
        if failed:
            raise RuntimeError(f"Could not load anonymization cascades: {', '.join(failed)}")

        self._cached_regions: list[Region] = []
        self._missed_refreshes = 0
        self._last_refresh_frame: int | None = None

    def anonymize(
        self,
        frame: np.ndarray,
        *,
        frame_index: int | None = None,
        force_refresh: bool = False,
        candidate_regions: Sequence[tuple[int, int, int, int]] | None = None,
        face_candidate_regions: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> np.ndarray:
        if frame is None:
            raise ValueError("frame is None")

        if force_refresh or self._should_refresh(frame_index):
            regions = _merge_regions(
                self._detect_regions(
                    frame,
                    candidate_regions=candidate_regions,
                    face_candidate_regions=face_candidate_regions,
                )
            )
            self._last_refresh_frame = frame_index
            if regions:
                self._cached_regions = regions
                self._missed_refreshes = 0
            else:
                self._missed_refreshes += 1
                if self._missed_refreshes > self.max_stale_cycles:
                    self._cached_regions = []

        return self._apply_blur(frame, self._cached_regions)

    def _should_refresh(self, frame_index: int | None) -> bool:
        if frame_index is None:
            return self._last_refresh_frame is None
        if self._last_refresh_frame is None:
            return True
        return (frame_index - self._last_refresh_frame) >= self.refresh_frames

    def _detect_regions(
        self,
        frame: np.ndarray,
        candidate_regions: Sequence[tuple[int, int, int, int]] | None = None,
        face_candidate_regions: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> list[Region]:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        frame_shape = gray.shape[:2]
        regions = self._detect_face_regions(gray, frame_shape=frame_shape, candidate_regions=face_candidate_regions)
        regions.extend(self._detect_plate_regions(gray, frame_shape=frame_shape, candidate_regions=candidate_regions))
        return _merge_regions(regions)

    def _detect_face_regions(
        self,
        gray: np.ndarray,
        *,
        frame_shape: tuple[int, int],
        candidate_regions: Sequence[tuple[int, int, int, int]] | None,
    ) -> list[Region]:
        if not candidate_regions:
            return self._detect_with_cascade(
                gray,
                self.face_cascade,
                scale_factor=self.face_scale_factor,
                min_neighbors=self.face_min_neighbors,
                min_size=self.face_min_size,
                frame_shape=frame_shape,
                min_aspect_ratio=0.75,
                max_aspect_ratio=1.6,
                max_area_ratio=0.08,
            )

        frame_area = max(1, frame_shape[0] * frame_shape[1])
        detections: list[Region] = []
        for candidate in candidate_regions:
            person_region = self._normalize_person_region(candidate, frame_shape)
            if person_region is None:
                continue
            roi = gray[person_region.y1:person_region.y2, person_region.x1:person_region.x2]
            if roi.size == 0:
                continue
            face_detections = self.face_cascade.detectMultiScale(
                roi,
                scaleFactor=max(1.05, self.face_scale_factor - 0.02),
                minNeighbors=max(3, self.face_min_neighbors - 1),
                minSize=(max(12, self.face_min_size[0] - 4), max(12, self.face_min_size[1] - 4)),
            )
            for x, y, w, h in face_detections:
                region = self._normalize_detection(
                    Region(
                        person_region.x1 + int(x),
                        person_region.y1 + int(y),
                        person_region.x1 + int(x + w),
                        person_region.y1 + int(y + h),
                    ),
                    frame_shape,
                    frame_area=frame_area,
                    min_aspect_ratio=0.65,
                    max_aspect_ratio=1.8,
                    max_area_ratio=0.03,
                )
                if region is not None:
                    detections.append(region)
        if detections:
            return detections
        return self._fallback_face_regions(candidate_regions, frame_shape)

    def _detect_plate_regions(
        self,
        gray: np.ndarray,
        *,
        frame_shape: tuple[int, int],
        candidate_regions: Sequence[tuple[int, int, int, int]] | None,
    ) -> list[Region]:
        if not candidate_regions:
            return self._detect_with_cascade(
                gray,
                self.plate_cascade,
                scale_factor=self.plate_scale_factor,
                min_neighbors=self.plate_min_neighbors,
                min_size=self.plate_min_size,
                frame_shape=frame_shape,
                min_aspect_ratio=1.1,
                max_aspect_ratio=7.0,
                max_area_ratio=0.04,
            )

        frame_area = max(1, frame_shape[0] * frame_shape[1])
        detections: list[Region] = []
        for candidate in candidate_regions:
            roi_region = self._expand_candidate_region(candidate, frame_shape)
            if roi_region is None:
                continue
            roi = gray[roi_region.y1:roi_region.y2, roi_region.x1:roi_region.x2]
            if roi.size == 0:
                continue
            roi_detections = self.plate_cascade.detectMultiScale(
                roi,
                scaleFactor=max(1.05, self.plate_scale_factor - 0.02),
                minNeighbors=max(2, self.plate_min_neighbors - 1),
                minSize=(max(12, self.plate_min_size[0]), max(8, self.plate_min_size[1])),
            )
            for x, y, w, h in roi_detections:
                region = self._normalize_detection(
                    Region(
                        roi_region.x1 + int(x),
                        roi_region.y1 + int(y),
                        roi_region.x1 + int(x + w),
                        roi_region.y1 + int(y + h),
                    ),
                    frame_shape,
                    frame_area=frame_area,
                    min_aspect_ratio=1.1,
                    max_aspect_ratio=7.0,
                    max_area_ratio=0.04,
                )
                if region is not None:
                    detections.append(region)
        if detections:
            return detections
        return self._fallback_plate_regions(candidate_regions, frame_shape)

    def _detect_with_cascade(
        self,
        gray: np.ndarray,
        cascade: cv2.CascadeClassifier,
        *,
        scale_factor: float,
        min_neighbors: int,
        min_size: tuple[int, int],
        frame_shape: tuple[int, int],
        min_aspect_ratio: float,
        max_aspect_ratio: float,
        max_area_ratio: float,
    ) -> list[Region]:
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        frame_area = max(1, frame_shape[0] * frame_shape[1])
        return [
            region
            for region in (
                self._normalize_detection(
                    Region(int(x), int(y), int(x + w), int(y + h)),
                    frame_shape,
                    frame_area=frame_area,
                    min_aspect_ratio=min_aspect_ratio,
                    max_aspect_ratio=max_aspect_ratio,
                    max_area_ratio=max_area_ratio,
                )
                for (x, y, w, h) in detections
            )
            if region is not None
        ]

    def _normalize_detection(
        self,
        region: Region,
        frame_shape: tuple[int, int],
        *,
        frame_area: int,
        min_aspect_ratio: float,
        max_aspect_ratio: float,
        max_area_ratio: float,
    ) -> Region | None:
        if region.width <= 0 or region.height <= 0:
            return None
        aspect_ratio = region.width / max(1, region.height)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            return None
        area_ratio = (region.width * region.height) / frame_area
        if area_ratio > max_area_ratio:
            return None
        return self._expand_region(region, frame_shape)

    def _expand_region(self, region: Region, frame_shape: tuple[int, int]) -> Region | None:
        expand_x = int(round(region.width * self.expand_ratio))
        expand_y = int(round(region.height * self.expand_ratio))
        expanded = Region(
            x1=region.x1 - expand_x,
            y1=region.y1 - expand_y,
            x2=region.x2 + expand_x,
            y2=region.y2 + expand_y,
        )
        return expanded.clipped(frame_shape)

    def _expand_candidate_region(
        self,
        candidate: Sequence[int],
        frame_shape: tuple[int, int],
    ) -> Region | None:
        if len(candidate) != 4:
            return None
        x1, y1, x2, y2 = (int(value) for value in candidate)
        region = Region(x1, y1, x2, y2)
        expand_x = int(round(region.width * 0.08))
        expand_y = int(round(region.height * 0.12))
        expanded = Region(
            x1=region.x1 - expand_x,
            y1=region.y1 - expand_y,
            x2=region.x2 + expand_x,
            y2=region.y2 + expand_y,
        )
        return expanded.clipped(frame_shape)

    def _fallback_plate_regions(
        self,
        candidate_regions: Sequence[tuple[int, int, int, int]],
        frame_shape: tuple[int, int],
    ) -> list[Region]:
        frame_area = max(1, frame_shape[0] * frame_shape[1])
        fallback_regions: list[Region] = []
        for candidate in candidate_regions:
            if len(candidate) != 4:
                continue
            vehicle = Region(*(int(value) for value in candidate))
            if vehicle.width <= 0 or vehicle.height <= 0:
                continue
            area_ratio = (vehicle.width * vehicle.height) / frame_area
            if area_ratio < 0.02 or area_ratio > 0.20:
                continue
            aspect_ratio = vehicle.width / max(1, vehicle.height)
            if aspect_ratio < 0.75 or aspect_ratio > 4.0:
                continue

            if vehicle.width >= vehicle.height:
                zones = (
                    Region(
                        vehicle.x1 + int(vehicle.width * 0.00),
                        vehicle.y1 + int(vehicle.height * 0.36),
                        vehicle.x1 + int(vehicle.width * 0.26),
                        vehicle.y1 + int(vehicle.height * 0.70),
                    ),
                    Region(
                        vehicle.x2 - int(vehicle.width * 0.26),
                        vehicle.y1 + int(vehicle.height * 0.36),
                        vehicle.x2 - int(vehicle.width * 0.00),
                        vehicle.y1 + int(vehicle.height * 0.70),
                    ),
                )
            else:
                zones = (
                    Region(
                        vehicle.x1 + int(vehicle.width * 0.24),
                        vehicle.y1 + int(vehicle.height * 0.00),
                        vehicle.x1 + int(vehicle.width * 0.76),
                        vehicle.y1 + int(vehicle.height * 0.24),
                    ),
                    Region(
                        vehicle.x1 + int(vehicle.width * 0.24),
                        vehicle.y2 - int(vehicle.height * 0.24),
                        vehicle.x1 + int(vehicle.width * 0.76),
                        vehicle.y2 - int(vehicle.height * 0.00),
                    ),
                )

            for zone in zones:
                expanded = self._expand_region(zone, frame_shape)
                if expanded is not None:
                    fallback_regions.append(expanded)
        return _merge_regions(fallback_regions)

    def _normalize_person_region(
        self,
        candidate: Sequence[int],
        frame_shape: tuple[int, int],
    ) -> Region | None:
        if len(candidate) != 4:
            return None
        person = Region(*(int(value) for value in candidate))
        frame_area = max(1, frame_shape[0] * frame_shape[1])
        area_ratio = (person.width * person.height) / frame_area
        if area_ratio < 0.001:
            return None
        if person.width <= 8 or person.height <= 16:
            return None
        return person.clipped(frame_shape)

    def _fallback_face_regions(
        self,
        candidate_regions: Sequence[tuple[int, int, int, int]],
        frame_shape: tuple[int, int],
    ) -> list[Region]:
        fallback_regions: list[Region] = []
        for candidate in candidate_regions:
            person = self._normalize_person_region(candidate, frame_shape)
            if person is None:
                continue
            head = Region(
                person.x1 + int(person.width * 0.18),
                person.y1 + int(person.height * 0.00),
                person.x1 + int(person.width * 0.82),
                person.y1 + int(person.height * 0.30),
            )
            expanded = self._expand_region(head, frame_shape)
            if expanded is not None:
                fallback_regions.append(expanded)
        return _merge_regions(fallback_regions)

    def _apply_blur(self, frame: np.ndarray, regions: list[Region]) -> np.ndarray:
        output = frame.copy()
        for region in regions:
            roi = output[region.y1:region.y2, region.x1:region.x2]
            if roi.size == 0:
                continue
            kernel = (
                _odd_kernel_size(roi.shape[1]),
                _odd_kernel_size(roi.shape[0]),
            )
            output[region.y1:region.y2, region.x1:region.x2] = cv2.GaussianBlur(roi, kernel, 0)
        return output
