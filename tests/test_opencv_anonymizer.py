from __future__ import annotations

import numpy as np

from smart_parking.anonymization.opencv import OpenCVHaarAnonymizer, Region


def _gradient_frame(height: int = 80, width: int = 80) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_anonymize_preserves_shape_dtype_and_limits_changes_to_roi(monkeypatch):
    anonymizer = OpenCVHaarAnonymizer(refresh_frames=1)
    frame = _gradient_frame()
    monkeypatch.setattr(anonymizer, "_detect_regions", lambda current, **kwargs: [Region(10, 10, 50, 50)])

    output = anonymizer.anonymize(frame, frame_index=1)

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype
    assert np.array_equal(output[0, 0], frame[0, 0])
    assert not np.array_equal(output[25, 25], frame[25, 25])


def test_anonymize_merges_overlapping_regions(monkeypatch):
    anonymizer = OpenCVHaarAnonymizer(refresh_frames=1)
    monkeypatch.setattr(
        anonymizer,
        "_detect_regions",
        lambda current, **kwargs: [Region(5, 5, 25, 25), Region(20, 10, 40, 30)],
    )

    anonymizer.anonymize(_gradient_frame(), frame_index=1)

    assert anonymizer._cached_regions == [Region(5, 5, 40, 30)]


def test_expand_region_clips_to_frame_bounds():
    anonymizer = OpenCVHaarAnonymizer(refresh_frames=1)

    expanded = anonymizer._expand_region(Region(-4, -6, 12, 10), (20, 20))

    assert expanded == Region(0, 0, 14, 12)


def test_cached_regions_expire_after_configured_refresh_misses(monkeypatch):
    anonymizer = OpenCVHaarAnonymizer(refresh_frames=1, max_stale_cycles=2)
    frame = _gradient_frame()
    detections = iter(
        [
            [Region(10, 10, 30, 30)],
            [],
            [],
            [],
        ]
    )
    monkeypatch.setattr(anonymizer, "_detect_regions", lambda current, **kwargs: next(detections))

    first = anonymizer.anonymize(frame, frame_index=1)
    second = anonymizer.anonymize(frame, frame_index=2)
    third = anonymizer.anonymize(frame, frame_index=3)
    fourth = anonymizer.anonymize(frame, frame_index=4)

    assert not np.array_equal(first[20, 20], frame[20, 20])
    assert not np.array_equal(second[20, 20], frame[20, 20])
    assert not np.array_equal(third[20, 20], frame[20, 20])
    assert np.array_equal(fourth[20, 20], frame[20, 20])


def test_fallback_plate_regions_ignore_tiny_boxes_and_create_bumper_zones():
    anonymizer = OpenCVHaarAnonymizer(refresh_frames=1)

    regions = anonymizer._fallback_plate_regions(
        [
            (75, 446, 351, 592),
            (0, 2, 64, 46),
        ],
        (739, 768),
    )

    assert len(regions) == 2
    assert all(region.width > 0 and region.height > 0 for region in regions)
    assert all(region.x1 >= 60 and region.x2 <= 370 for region in regions)
