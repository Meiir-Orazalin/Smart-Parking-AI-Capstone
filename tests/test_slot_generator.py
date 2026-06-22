from __future__ import annotations

import pytest

from smart_parking.slots.generator import SeedValidationError, generate_slots_from_seed_payload


def _center(slot: dict) -> tuple[float, float]:
    points = slot["points"]
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _normalized_edge(points: list[list[int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    left = tuple(points[0])
    right = tuple(points[1])
    return tuple(sorted((left, right)))


def test_generate_slots_interpolates_missing_slots_between_anchors():
    payload = {
        "image_shape": [100, 100],
        "anchors": [
            {
                "row": 1,
                "slot_index": 1,
                "slot_count": 3,
                "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
            },
            {
                "row": 1,
                "slot_index": 3,
                "slot_count": 3,
                "points": [[30, 30], [44, 30], [44, 42], [30, 42]],
            },
        ],
    }

    slots = generate_slots_from_seed_payload(payload)

    assert len(slots) == 3
    assert slots[0]["id"] == "r01s01"
    assert slots[1]["id"] == "r01s02"
    assert slots[2]["id"] == "r01s03"
    assert _normalized_edge(slots[0]["points"][2:]) == _normalized_edge(slots[1]["points"][:2])
    assert _normalized_edge(slots[1]["points"][2:]) == _normalized_edge(slots[2]["points"][:2])


def test_generate_slots_normalizes_anchor_corners_before_generation():
    payload = {
        "image_shape": [120, 120],
        "anchors": [
            {
                "row": 1,
                "slot_index": 1,
                "slot_count": 3,
                "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
            },
            {
                "row": 1,
                "slot_index": 3,
                "slot_count": 3,
                "points": [[30, 42], [44, 42], [44, 30], [30, 30]],
            },
        ],
    }

    slots = generate_slots_from_seed_payload(payload)

    slot0_points = slots[0]["points"]
    assert len(slot0_points) == 4
    assert len({tuple(point) for point in slot0_points}) == 4
    assert _normalized_edge(slots[0]["points"][2:]) == _normalized_edge(slots[1]["points"][:2])


def test_generate_slots_uses_middle_anchors_piecewise():
    payload = {
        "image_shape": [140, 140],
        "anchors": [
            {
                "row": 1,
                "slot_index": 1,
                "slot_count": 4,
                "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
            },
            {
                "row": 1,
                "slot_index": 2,
                "slot_count": 4,
                "points": [[18, 18], [32, 18], [32, 30], [18, 30]],
            },
            {
                "row": 1,
                "slot_index": 4,
                "slot_count": 4,
                "points": [[42, 42], [62, 42], [62, 60], [42, 60]],
            },
        ],
    }

    slots = generate_slots_from_seed_payload(payload)

    assert len(slots) == 4
    assert slots[2]["id"] == "r01s03"
    centers = [_center(slot) for slot in slots]
    assert centers[0][1] < centers[1][1] < centers[2][1] < centers[3][1]
    assert _normalized_edge(slots[1]["points"][2:]) == _normalized_edge(slots[2]["points"][:2])


def test_generate_slots_requires_first_and_last_anchor_in_each_row():
    payload = {
        "anchors": [
            {
                "row": 1,
                "slot_index": 2,
                "slot_count": 4,
                "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
            },
            {
                "row": 1,
                "slot_index": 4,
                "slot_count": 4,
                "points": [[30, 30], [40, 30], [40, 40], [30, 40]],
            },
        ]
    }

    with pytest.raises(SeedValidationError, match="first and last slot"):
        generate_slots_from_seed_payload(payload)
