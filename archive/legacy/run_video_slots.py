import argparse
import cv2
import numpy as np

from smart_parking_detection import SmartParkingV2


def main():
    parser = argparse.ArgumentParser(description="Run slot occupancy on a video.")
    parser.add_argument(
        "--video",
        default="ParkingVideo.mov",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=12,
        help="Number of frames to skip between processed frames.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.6,
        help="Scale factor for frames (smaller is faster).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="YOLO inference image size.",
    )
    args = parser.parse_args()

    video_path = args.video
    slots_path = "slots.json"

    # Tuned demo settings
    conf = 0.01
    overlap = 0.26
    imgsz = args.imgsz
    pad = 0.03
    unsure = 0.55
    smooth_frames = 5
    display_scale = args.display_scale
    frame_skip = args.frame_skip

    detector = SmartParkingV2()
    slots = detector._load_slots(slots_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Temporal smoothing state
    states = [{"state": None, "streak": 0} for _ in range(len(slots))]

    frame_idx = 0
    last_out = None
    last_slot_list = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # Optional downscale for speed
        orig_h, orig_w = frame.shape[:2]
        if display_scale != 1.0:
            frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

            # Scale slots to match resized frame
            h, w = frame.shape[:2]
            sx = w / orig_w if orig_w else 1.0
            sy = h / orig_h if orig_h else 1.0
            scaled_slots = []
            for s in slots:
                pts = []
                for (x, y) in s["points"]:
                    pts.append([int(x * sx), int(y * sy)])
                scaled_slots.append({"points": pts})
            slots_for_frame = scaled_slots
        else:
            slots_for_frame = slots

        process_this_frame = frame_skip <= 0 or (frame_idx % (frame_skip + 1)) == 1

        if process_this_frame:
            _, slot_list, *_ = detector.detect_with_slots_frame(
                frame,
                slots_for_frame,
                conf_threshold=conf,
                overlap_threshold=overlap,
                imgsz=imgsz,
                pad_ratio=pad,
                unsure_threshold=unsure,
            )

            # Smooth occupancy labels across frames (avoid flicker)
            for i, s in enumerate(slot_list):
                raw = 1 if s["occupied"] else 0
                st = states[i]
                if st["state"] is None:
                    st["state"] = raw
                    st["streak"] = 0
                elif raw == st["state"]:
                    st["streak"] = 0
                else:
                    st["streak"] += 1
                    if st["streak"] >= smooth_frames:
                        st["state"] = raw
                        st["streak"] = 0

            last_slot_list = slot_list

        out = frame.copy()
        if last_slot_list is not None:
            occupied_count = 0
            uncertain_count = 0
            # Redraw overlays with smoothed state (reduces flicker)
            for i, s in enumerate(last_slot_list):
                state = states[i]["state"]
                if state is None:
                    continue
                pts = s["points"]
                pts_np = np.array(pts, dtype=np.int32)

                if s.get("unsure"):
                    color = (0, 255, 255)
                    label = f"#{i+1} UNSURE"
                    uncertain_count += 1
                else:
                    is_occupied = state == 1
                    color = (0, 0, 255) if is_occupied else (0, 255, 0)
                    label = f"#{i+1} {'OCCUPIED' if is_occupied else 'AVAILABLE'}"
                    if is_occupied:
                        occupied_count += 1

                cv2.polylines(out, [pts_np], True, color, 3)
                overlay = out.copy()
                cv2.fillPoly(overlay, [pts_np], color)
                out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
                centroid = (int(sum(p[0] for p in pts) / len(pts)), int(sum(p[1] for p in pts) / len(pts)))
                cv2.putText(out, label, (centroid[0] - 50, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                conf_val = s.get("confidence", 0)
                if conf_val > 0:
                    conf_text = f"{conf_val*100:.1f}%"
                    cv2.putText(out, conf_text, (centroid[0] - 20, centroid[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            total_slots = len(last_slot_list)
            detector._draw_slot_stats(out, total_slots, occupied_count, uncertain_count)

        last_out = out
        cv2.imshow("Smart Parking - Video", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
