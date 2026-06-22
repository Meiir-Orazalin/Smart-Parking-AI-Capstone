from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from smart_parking.app.video_demo import _compose_output_frame, _register_view
from smart_parking.occupancy import OccupancyResult
from smart_parking.slots import load_view_cache


class StubAnonymizer:
    backend_name = "stub"

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.force_refresh_calls: list[bool] = []
        self.candidate_regions: list[tuple[tuple[int, int, int, int], ...]] = []
        self.face_candidate_regions: list[tuple[tuple[int, int, int, int], ...]] = []

    def anonymize(
        self,
        frame: np.ndarray,
        *,
        frame_index: int | None = None,
        force_refresh: bool = False,
        candidate_regions: tuple[tuple[int, int, int, int], ...] | None = None,
        face_candidate_regions: tuple[tuple[int, int, int, int], ...] | None = None,
    ) -> np.ndarray:
        self.frames.append(frame.copy())
        self.force_refresh_calls.append(force_refresh)
        self.candidate_regions.append(tuple(candidate_regions or ()))
        self.face_candidate_regions.append(tuple(face_candidate_regions or ()))
        output = np.full_like(frame, 240)
        output[0, 0] = (1, 2, 3)
        return output


def test_compose_output_frame_runs_anonymization_after_occupancy_overlay():
    work_frame = np.full((240, 320, 3), 10, dtype=np.uint8)
    annotated = np.full((240, 320, 3), 80, dtype=np.uint8)
    last_result = OccupancyResult(
        total_slots=1,
        occupied=1,
        available=0,
        unsure=0,
        slots=tuple(),
        vehicle_boxes=((10, 20, 70, 80),),
        person_boxes=((100, 40, 130, 110),),
        annotated_frame=annotated,
    )
    anonymizer = StubAnonymizer()

    def overlay(frame, statuses):
        result = frame.copy()
        result[210, 280] = (80, 80, 80)
        return result

    output, capacity, occupied, available, unsure = _compose_output_frame(
        work_frame,
        last_result,
        process_this_frame=True,
        annotate_statuses=overlay,
        slot_mode="manual",
        view_status="ready",
        active_view_id=None,
        calibration_status="ready",
        frame_index=3,
        anonymizer=anonymizer,
    )

    assert (capacity, occupied, available, unsure) == (1, 1, 0, 0)
    assert anonymizer.frames[0][210, 280].tolist() == [10, 10, 10]
    assert anonymizer.candidate_regions == [((10, 20, 70, 80),)]
    assert anonymizer.face_candidate_regions == [((100, 40, 130, 110),)]
    assert output[210, 280].tolist() == [80, 80, 80]
    assert output[230, 310].tolist() == [240, 240, 240]


def test_register_view_saves_anonymized_reference_image(tmp_path: Path):
    cache_path = tmp_path / "auto_slots.json"
    cache = load_view_cache(cache_path)
    frame = np.full((40, 40, 3), 25, dtype=np.uint8)
    slots = [{"points": [[1, 1], [10, 1], [10, 10], [1, 10]]}]
    anonymizer = StubAnonymizer()

    view = _register_view(cache_path, cache, frame, slots, anonymizer=anonymizer)

    saved_path = cache_path.parent / view.reference_frame_path
    saved = cv2.imread(str(saved_path))

    assert anonymizer.force_refresh_calls == [True]
    assert saved is not None
    assert float(saved.mean()) > 200.0
