from __future__ import annotations

import numpy as np

from smart_parking.detection.manual_detector import SmartParkingV2
from smart_parking.occupancy import OccupancyEngine


def test_slot_dynamic_threshold_uses_fixed_value_when_disabled():
    points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)

    threshold = SmartParkingV2._slot_dynamic_threshold(
        points,
        (100, 100),
        base_threshold=0.55,
        enabled=False,
        near_threshold=0.70,
        far_threshold=0.20,
    )

    assert threshold == 0.55


def test_slot_dynamic_threshold_is_lower_for_far_slots_and_higher_for_near_slots():
    far_points = np.array([[0, 5], [10, 5], [10, 15], [0, 15]], dtype=np.int32)
    near_points = np.array([[0, 85], [10, 85], [10, 95], [0, 95]], dtype=np.int32)

    far_threshold = SmartParkingV2._slot_dynamic_threshold(
        far_points,
        (100, 100),
        enabled=True,
        near_threshold=0.55,
        far_threshold=0.30,
    )
    near_threshold = SmartParkingV2._slot_dynamic_threshold(
        near_points,
        (100, 100),
        enabled=True,
        near_threshold=0.55,
        far_threshold=0.30,
    )

    assert 0.30 < far_threshold < near_threshold < 0.55


class CapturingBackend:
    def __init__(self) -> None:
        self.kwargs = {}

    def detect_with_slots_frame(self, frame, slots, **kwargs):
        self.kwargs = kwargs
        raw_statuses = [
            {
                "occupied": True,
                "unsure": False,
                "confidence": 0.60,
                "overlap_threshold": 0.26,
                "occupancy_threshold": 0.55,
            }
        ]
        return frame.copy(), raw_statuses, 1, 0, 0, [], []


def test_occupancy_engine_passes_dynamic_threshold_options_to_backend():
    backend = CapturingBackend()
    engine = OccupancyEngine(
        backend=backend,
        dynamic_occupancy_threshold=True,
        near_occupancy_threshold=0.60,
        far_occupancy_threshold=0.25,
        dynamic_threshold_axis="y",
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    slots = [{"points": [[0, 0], [20, 0], [20, 20], [0, 20]]}]

    result = engine.process_frame(frame, slots)

    assert result.occupied == 1
    assert backend.kwargs["dynamic_occupancy_threshold"] is True
    assert backend.kwargs["near_occupancy_threshold"] == 0.60
    assert backend.kwargs["far_occupancy_threshold"] == 0.25
    assert backend.kwargs["dynamic_threshold_axis"] == "y"
