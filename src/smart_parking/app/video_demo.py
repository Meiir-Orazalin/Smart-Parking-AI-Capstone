from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import uvicorn

from smart_parking.anonymization import OpenCVHaarAnonymizer
from smart_parking.api import video_backend
from smart_parking.occupancy import OccupancyEngine
from smart_parking.slots import (
    FrameStabilityTracker,
    ORBViewMatcher,
    SlotView,
    ViewStateManager,
    YOLOSlotPolygonDetector,
    generate_slots_from_seed_file,
    load_anchor_seed_file,
    load_slot_file,
    load_view_cache,
    save_view_cache,
)
from smart_parking.utils.paths import (
    default_manual_slots_path,
    default_slot_model_path,
    default_slots_cache_path,
    default_video_path,
    resolve_repo_path,
)


def _is_stream_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "rtsp://"))


def start_api(host: str, port: int) -> None:
    config = uvicorn.Config(video_backend.app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the smart parking video demo.")
    parser.add_argument("--video", default=str(default_video_path()), help="Input video file path.")
    parser.add_argument("--slot-mode", choices=("auto", "manual"), default="auto", help="Slot source strategy.")
    parser.add_argument("--slots", default=str(default_manual_slots_path()), help="Manual slot JSON path.")
    parser.add_argument("--anchor-seeds", default="", help="Anchor seed JSON path used to generate manual slots at startup.")
    parser.add_argument("--slots-cache", default=str(default_slots_cache_path()), help="Auto slot cache path.")
    parser.add_argument("--slot-model", default=str(default_slot_model_path()), help="Segmentation model for auto slot detection.")
    parser.add_argument("--slots-source-image", default="", help="Image used to define manual slots, if different from the video frame shape.")
    parser.add_argument("--calib-frames", type=int, default=12, help="Stable frames required before auto calibration.")
    parser.add_argument("--frame-skip", type=int, default=12, help="Frames to skip between occupancy inference.")
    parser.add_argument("--display-scale", type=float, default=0.6, help="Scale factor for displayed frames.")
    parser.add_argument("--imgsz", type=int, default=480, help="Vehicle detector image size.")
    parser.add_argument("--unsure", type=float, default=0.55, help="Unsure threshold (0-1).")
    parser.add_argument("--conf", type=float, default=0.01, help="Vehicle detector confidence threshold.")
    parser.add_argument("--overlap", type=float, default=0.26, help="Slot overlap threshold.")
    parser.add_argument("--dynamic-threshold", action="store_true", help="Vary occupied/unsure confidence threshold by perspective position.")
    parser.add_argument("--near-threshold", type=float, default=0.55, help="Occupied threshold for near/bottom slots in dynamic mode.")
    parser.add_argument("--far-threshold", type=float, default=0.30, help="Occupied threshold for far/top slots in dynamic mode.")
    parser.add_argument("--dynamic-axis", choices=("x", "y"), default="y", help="Image axis used for dynamic threshold interpolation.")
    parser.add_argument("--show-thresholds", action="store_true", help="Show measured confidence and occupied threshold on occupied/unsure slots.")
    parser.add_argument("--pad", type=float, default=0.03, help="Padding ratio for vehicle boxes.")
    parser.add_argument("--smooth", type=int, default=5, help="Frames to confirm state change.")
    parser.add_argument("--motion-threshold", type=float, default=6.0, help="Mean grayscale delta treated as stable motion.")
    parser.add_argument("--match-score", type=float, default=0.20, help="Minimum ORB view-match score.")
    parser.add_argument("--min-auto-slots", type=int, default=4, help="Reject auto-calibrated views with fewer slots.")
    parser.add_argument("--max-auto-slots", type=int, default=400, help="Reject auto-calibrated views with too many slots.")
    parser.add_argument("--no-anonymize", action="store_true", help="Disable face and license-plate anonymization.")
    parser.add_argument("--anonymize-refresh-frames", type=int, default=3, help="Frames between anonymization detection refreshes.")
    parser.add_argument("--api-host", default="127.0.0.1", help="API host.")
    parser.add_argument("--api-port", type=int, default=8000, help="API port.")
    return parser.parse_args(argv)


def _source_image_shape(path: str) -> tuple[int, int] | None:
    if not path:
        return None
    image = cv2.imread(str(resolve_repo_path(path)))
    if image is None:
        return None
    return image.shape[:2]


def _ensure_payload_shape(payload: dict | None, image_shape: tuple[int, int] | None) -> dict | None:
    if payload is None:
        return None
    result = dict(payload)
    if result.get("image_shape") is None and image_shape is not None:
        result["image_shape"] = list(image_shape)
    return result


def _draw_stats_panel(frame: np.ndarray, occupied: int, available: int, unsure: int, total: int) -> None:
    height, width = frame.shape[:2]
    panel_width = 250
    panel_height = 180
    panel_x = 10
    panel_y = 10
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 180, 255), 2)

    occupancy_pct = (occupied / total * 100.0) if total > 0 else 0.0
    lines = [
        ("SMART PARKING", (0, 180, 255), 0.6, 24),
        (f"Total Slots: {total}", (255, 255, 255), 0.45, 58),
        (f"Occupied: {occupied}", (0, 0, 255), 0.45, 82),
        (f"Available: {available}", (0, 255, 0), 0.45, 106),
        (f"Unsure: {unsure}", (0, 255, 255), 0.45, 130),
        (f"Occupancy: {occupancy_pct:.1f}%", (255, 255, 255), 0.45, 154),
    ]
    for text, color, scale, y in lines:
        cv2.putText(frame, text, (panel_x + 10, panel_y + y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if scale < 0.5 else 2)


def _draw_runtime_banner(
    frame: np.ndarray,
    *,
    slot_mode: str,
    view_status: str,
    active_view_id: str | None,
    calibration_status: str,
) -> None:
    text = f"mode={slot_mode} status={view_status} calib={calibration_status}"
    if active_view_id:
        text += f" view={active_view_id}"
    cv2.putText(
        frame,
        text,
        (10, max(24, frame.shape[0] - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )


def _build_payload_from_view(view: SlotView | None) -> dict | None:
    if view is None:
        return None
    return {
        "image_shape": list(view.image_shape) if view.image_shape is not None else None,
        "slots": view.slots,
    }


def _register_view(
    cache_path: Path,
    cache,
    frame: np.ndarray,
    slots: list[dict],
    *,
    anonymizer: Any | None = None,
) -> SlotView:
    reference_dir = cache_path.with_suffix("")
    reference_dir.mkdir(parents=True, exist_ok=True)
    view_id = f"view-{len(cache.views) + 1:04d}"
    reference_path = reference_dir / f"{view_id}.jpg"
    frame_to_save = anonymizer.anonymize(frame, force_refresh=True) if anonymizer is not None else frame
    cv2.imwrite(str(reference_path), frame_to_save)

    view = SlotView(
        id=view_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        image_shape=frame.shape[:2],
        reference_frame_path=str(reference_path.relative_to(cache_path.parent)),
        slots=slots,
    )
    cache.upsert_view(view)
    save_view_cache(cache_path, cache)
    return view


def _compose_output_frame(
    work_frame: np.ndarray,
    last_result,
    *,
    process_this_frame: bool,
    annotate_statuses: Callable[[np.ndarray, Any], np.ndarray],
    slot_mode: str,
    view_status: str,
    active_view_id: str | None,
    calibration_status: str,
    frame_index: int,
    anonymizer: Any | None = None,
) -> tuple[np.ndarray, int, int, int, int]:
    candidate_regions = tuple(last_result.vehicle_boxes) if last_result is not None else tuple()
    face_candidate_regions = tuple(last_result.person_boxes) if last_result is not None else tuple()
    output = (
        anonymizer.anonymize(
            work_frame,
            frame_index=frame_index,
            candidate_regions=candidate_regions,
            face_candidate_regions=face_candidate_regions,
        )
        if anonymizer is not None
        else work_frame.copy()
    )
    if last_result is not None:
        output = annotate_statuses(output, last_result.slots)
        _draw_stats_panel(output, last_result.occupied, last_result.available, last_result.unsure, last_result.total_slots)
        capacity = last_result.total_slots
        occupied = last_result.occupied
        available = last_result.available
        unsure = last_result.unsure
    else:
        capacity = 0
        occupied = 0
        available = 0
        unsure = 0

    _draw_runtime_banner(
        output,
        slot_mode=slot_mode,
        view_status=view_status,
        active_view_id=active_view_id,
        calibration_status=calibration_status,
    )
    return output, capacity, occupied, available, unsure


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    video_source = args.video if _is_stream_url(args.video) else str(resolve_repo_path(args.video))
    manual_slots_path = resolve_repo_path(args.slots)
    anchor_seeds_path = resolve_repo_path(args.anchor_seeds) if args.anchor_seeds else None
    slots_cache_path = resolve_repo_path(args.slots_cache)
    slot_model_path = resolve_repo_path(args.slot_model)
    slot_source_shape = _source_image_shape(args.slots_source_image)

    if not _is_stream_url(video_source) and not Path(video_source).exists():
        raise FileNotFoundError(f"Video not found: {video_source}")

    os.environ["BACKEND_NO_LOOP"] = "1"

    manual_slots_payload = None
    if anchor_seeds_path is not None and not anchor_seeds_path.exists():
        raise FileNotFoundError(f"Anchor seed file not found: {anchor_seeds_path}")

    if anchor_seeds_path is None and manual_slots_path.exists():
        manual_slots_payload = load_slot_file(manual_slots_path)
        if manual_slots_payload.get("version") == 1 and "views" in manual_slots_payload:
            manual_slots_payload = None

    anonymization_enabled = not args.no_anonymize
    anonymizer = None
    if anonymization_enabled:
        anonymizer = OpenCVHaarAnonymizer(refresh_frames=args.anonymize_refresh_frames)

    auto_detector = None
    if args.slot_mode == "auto" and slot_model_path.exists():
        auto_detector = YOLOSlotPolygonDetector(slot_model_path)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_source}")

    # Seed initial frame shape for manual slots if the source image is unknown.
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"No frames read from video: {video_source}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    original_frame_shape = first_frame.shape[:2]
    if anchor_seeds_path is not None:
        seed_payload = load_anchor_seed_file(anchor_seeds_path)
        generated_slots = generate_slots_from_seed_file(anchor_seeds_path)
        manual_slots_payload = {
            "image_shape": seed_payload.get("image_shape") or list(original_frame_shape),
            "slots": generated_slots,
        }
    if slot_source_shape is None:
        slot_source_shape = original_frame_shape
    manual_slots_payload = _ensure_payload_shape(manual_slots_payload, slot_source_shape)

    video_backend.state["video_path"] = video_source
    video_backend.state["slot_mode"] = args.slot_mode
    video_backend.state["view_status"] = "warming_up" if args.slot_mode == "auto" else "ready"
    video_backend.state["cache_ready"] = bool(slots_cache_path.exists())
    video_backend.state["calibration_status"] = "pending" if args.slot_mode == "auto" else "ready"
    video_backend.state["anonymization_enabled"] = anonymization_enabled
    video_backend.state["anonymization_backend"] = anonymizer.backend_name if anonymizer is not None else "disabled"
    video_backend.state["anonymization_refresh_frames"] = args.anonymize_refresh_frames if anonymization_enabled else 0

    api_thread = None
    if args.api_port > 0:
        import threading

        api_thread = threading.Thread(target=start_api, args=(args.api_host, args.api_port), daemon=True)
        api_thread.start()

    occupancy = OccupancyEngine(
        conf_threshold=args.conf,
        overlap_threshold=args.overlap,
        imgsz=args.imgsz,
        pad_ratio=args.pad,
        unsure_threshold=args.unsure,
        smooth_frames=args.smooth,
        dynamic_occupancy_threshold=args.dynamic_threshold,
        near_occupancy_threshold=args.near_threshold,
        far_occupancy_threshold=args.far_threshold,
        dynamic_threshold_axis=args.dynamic_axis,
        show_thresholds=args.show_thresholds,
    )

    cache = load_view_cache(slots_cache_path)
    matcher = ORBViewMatcher(min_score=args.match_score)
    view_state = ViewStateManager(
        cache,
        matcher=matcher,
        stable_motion_threshold=args.motion_threshold,
        stable_frames_required=max(2, args.calib_frames),
    )
    stability = FrameStabilityTracker(
        min_stable_frames=max(2, args.calib_frames),
        motion_threshold=args.motion_threshold,
    )

    frame_idx = 0
    last_fps_time = time.time()
    last_fps_count = 0
    fps_estimate = 0.0
    last_calibration_attempt = -10_000
    last_result = None

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(video_source)
            time.sleep(0.05)
            continue

        frame_idx += 1
        now = time.time()
        last_fps_count += 1
        if now - last_fps_time >= 1.0:
            fps_estimate = last_fps_count / max(0.001, now - last_fps_time)
            last_fps_time = now
            last_fps_count = 0

        work_frame = frame
        if args.display_scale != 1.0:
            work_frame = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)

        stability.update(work_frame)
        current_slot_mode = args.slot_mode
        view_status = "ready" if args.slot_mode == "manual" else "warming_up"
        calibration_status = "ready" if args.slot_mode == "manual" else "pending"
        active_view_id = None

        slot_payload = None
        if args.slot_mode == "manual":
            slot_payload = manual_slots_payload
        else:
            snapshot = view_state.observe(work_frame)
            view_status = snapshot.state
            active_view_id = snapshot.active_view_id

            if snapshot.active_view_id is not None:
                slot_payload = _build_payload_from_view(view_state.cache.get_view(snapshot.active_view_id))
                calibration_status = "ready"

            should_attempt_calibration = (
                snapshot.should_calibrate
                and stability.is_stable
                and auto_detector is not None
                and (frame_idx - last_calibration_attempt) >= max(2, args.calib_frames)
            )
            if should_attempt_calibration:
                last_calibration_attempt = frame_idx
                calibration_status = "running"
                reference_frame = stability.reference_frame()
                calibration_frame = reference_frame if reference_frame is not None else work_frame
                try:
                    detected_slots = auto_detector.detect_slots(calibration_frame)
                except Exception:
                    detected_slots = []

                if args.min_auto_slots <= len(detected_slots) <= args.max_auto_slots:
                    view = _register_view(
                        slots_cache_path,
                        cache,
                        calibration_frame,
                        detected_slots,
                        anonymizer=anonymizer,
                    )
                    cache = load_view_cache(slots_cache_path)
                    view_state.set_cache(cache)
                    view_state.set_active_view(view.id)
                    active_view_id = view.id
                    slot_payload = _build_payload_from_view(view_state.cache.get_view(view.id))
                    view_status = "ready"
                    calibration_status = "ready"
                else:
                    calibration_status = "failed"

            if slot_payload is None and manual_slots_payload is not None:
                current_slot_mode = "manual_fallback"
                slot_payload = manual_slots_payload
                if calibration_status == "pending":
                    calibration_status = "manual_fallback"

        process_this_frame = args.frame_skip <= 0 or (frame_idx % (args.frame_skip + 1)) == 1
        if slot_payload is not None and process_this_frame:
            last_result = occupancy.process_frame(work_frame, slot_payload, annotate=True)
        elif slot_payload is None:
            last_result = None

        output, capacity, occupied, available, unsure = _compose_output_frame(
            work_frame,
            last_result,
            process_this_frame=process_this_frame,
            annotate_statuses=occupancy.annotate_statuses,
            slot_mode=current_slot_mode,
            view_status=view_status,
            active_view_id=active_view_id,
            calibration_status=calibration_status,
            frame_index=frame_idx,
            anonymizer=anonymizer,
        )

        video_backend.update_state(
            frame_time=now,
            occupied=occupied,
            available=available,
            unsure=unsure,
            capacity=capacity,
            fps_estimate=fps_estimate,
            video_path=video_source,
            slot_mode=current_slot_mode,
            active_view_id=active_view_id,
            view_status=view_status,
            cache_ready=bool(cache.views),
            calibration_status=calibration_status,
            anonymization_enabled=anonymization_enabled,
            anonymization_backend=anonymizer.backend_name if anonymizer is not None else "disabled",
            anonymization_refresh_frames=args.anonymize_refresh_frames if anonymization_enabled else 0,
        )

        cv2.imshow("Smart Parking - Video", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
