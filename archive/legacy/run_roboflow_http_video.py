import os
import time
import threading

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient


def main():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY env var before running.")

    model_id = os.getenv("ROBOFLOW_MODEL_ID", "parking-spaces-ezhxz/1")
    video_path = os.getenv("VIDEO_PATH", "CarPark.mp4")
    roi_slots_path = os.getenv("ROI_SLOTS_PATH", "slots.json")
    slots_source_image = os.getenv("SLOTS_SOURCE_IMAGE")

    frame_skip = int(os.getenv("FRAME_SKIP", "0"))
    max_fps = float(os.getenv("MAX_FPS", "20"))
    scale = float(os.getenv("SCALE", "1.0"))
    jpeg_quality = int(os.getenv("JPEG_QUALITY", "90"))
    infer_every_sec = float(os.getenv("INFER_EVERY_SEC", "2.0"))
    occ_min_conf = float(os.getenv("OCC_MIN_CONF", "0.3"))
    slot_occ_ioa = float(os.getenv("SLOT_OCCUPIED_IOA", "0.2"))
    draw_slots = os.getenv("DRAW_SLOTS", "1") == "1"
    draw_pred_boxes = os.getenv("DRAW_PRED_BOXES", "0") == "1"

    client = InferenceHTTPClient.init(
        api_url="https://detect.roboflow.com",
        api_key=api_key,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    loop_video = os.getenv("LOOP_VIDEO", "1") == "1"
    override_fps = os.getenv("REALTIME_FPS")
    source_fps = float(override_fps) if override_fps else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_delay = 1.0 / source_fps if source_fps > 0 else 1.0 / 30.0
    delay_ms = max(1, int(frame_delay * 1000))
    print(f"Playback FPS target: {source_fps:.2f} (delay {delay_ms} ms), frame_skip={frame_skip}")

    # Build ROI polygon + scaled slot polygons from slots.json if available
    roi_polygon = None
    slots_scaled = []
    slots_scale_x = 1.0
    slots_scale_y = 1.0
    if slots_source_image and os.path.exists(slots_source_image):
        try:
            from PIL import Image

            img = Image.open(slots_source_image)
            src_w, src_h = img.size
            vid_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            vid_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if src_w > 0 and src_h > 0 and vid_w > 0 and vid_h > 0:
                slots_scale_x = vid_w / src_w
                slots_scale_y = vid_h / src_h
                print(
                    f"Slots scaling: source={src_w}x{src_h} -> video={int(vid_w)}x{int(vid_h)} "
                    f"scale=({slots_scale_x:.3f}, {slots_scale_y:.3f})"
                )
        except Exception:
            pass

    if os.path.exists(roi_slots_path):
        try:
            import json
            with open(roi_slots_path, "r") as f:
                slots = json.load(f)
            all_pts = []
            for s in slots:
                pts = s.get("points", [])
                if len(pts) < 3:
                    continue
                pts_scaled = np.array(
                    [
                        [p[0] * slots_scale_x * scale, p[1] * slots_scale_y * scale]
                        for p in pts
                    ],
                    dtype=np.float32,
                )
                # Ensure a valid, ordered convex polygon for drawing/intersection
                hull = cv2.convexHull(pts_scaled)
                area = float(cv2.contourArea(hull))
                if area <= 1.0:
                    continue
                slots_scaled.append({"points": hull, "area": area})
                for p in hull:
                    all_pts.append(p)
            if all_pts:
                all_pts = np.array(all_pts, dtype=np.float32)
                roi_polygon = cv2.convexHull(all_pts, returnPoints=True)
        except Exception:
            roi_polygon = None

    frame_idx = 0
    last_time = 0.0
    last_preds = []
    latest_frame = None
    frame_lock = threading.Lock()
    stop_flag = {"stop": False}

    def infer_worker():
        nonlocal last_preds, latest_frame
        last_infer_time = 0.0
        while not stop_flag["stop"]:
            now = time.time()
            if now - last_infer_time < infer_every_sec:
                time.sleep(0.01)
                continue

            with frame_lock:
                if latest_frame is None:
                    frame_to_infer = None
                else:
                    frame_to_infer = latest_frame.copy()

            if frame_to_infer is None:
                time.sleep(0.01)
                continue

            # Downscale for speed (optional)
            if scale != 1.0:
                frame_small = cv2.resize(frame_to_infer, None, fx=scale, fy=scale)
            else:
                frame_small = frame_to_infer

            tmp_path = "rf_frame.jpg"
            cv2.imwrite(tmp_path, frame_small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            result = client.infer(tmp_path, model_id=model_id)
            last_preds = result.get("predictions", [])
            last_infer_time = now

    t = threading.Thread(target=infer_worker, daemon=True)
    t.start()

    while True:
        start_tick = time.time()
        ok, frame = cap.read()
        if not ok:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_idx += 1
        if frame_skip > 0 and (frame_idx % (frame_skip + 1)) != 1:
            continue

        # Throttle to max_fps
        now = time.time()
        if max_fps > 0 and now - last_time < (1.0 / max_fps):
            continue
        last_time = now

        # Downscale for speed (optional)
        if scale != 1.0:
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            frame_small = frame

        # Update latest frame for inference thread
        with frame_lock:
            latest_frame = frame

        # Draw cached predictions onto current frame for smooth video
        out = frame_small.copy()
        occ_boxes = []
        for p in last_preds:
            x = int(p["x"])
            y = int(p["y"])
            w = int(p["width"])
            h = int(p["height"])
            label = p.get("class", "")
            conf = float(p.get("confidence", 0.0))
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            if roi_polygon is not None:
                if cv2.pointPolygonTest(roi_polygon, (x, y), False) < 0:
                    continue
            if label.lower() == "occupied" and conf >= occ_min_conf:
                occ_boxes.append((x1, y1, x2, y2, conf))
            if draw_pred_boxes:
                color = (0, 0, 255) if label.lower() == "occupied" else (0, 255, 0)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(
                    out,
                    text,
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        # Slot-level occupancy: mark slot occupied if it overlaps any occupied box
        if draw_slots and slots_scaled:
            occupied_count = 0
            for slot in slots_scaled:
                pts = slot["points"]
                slot_area = slot["area"]
                is_occ = False
                for (x1, y1, x2, y2, _) in occ_boxes:
                    rect = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32,
                    )
                    inter_area, _ = cv2.intersectConvexConvex(pts, rect)
                    if slot_area > 0 and (inter_area / slot_area) >= slot_occ_ioa:
                        is_occ = True
                        break
                color = (0, 0, 255) if is_occ else (0, 255, 0)
                cv2.polylines(out, [pts.astype(np.int32)], True, color, 2)
                if is_occ:
                    occupied_count += 1

            total_slots = len(slots_scaled)
            empty_slots = total_slots - occupied_count
            status_text = f"Total: {total_slots}  Occupied: {occupied_count}  Empty: {empty_slots}"
            cv2.putText(
                out,
                status_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Roboflow HTTP Video", out)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            stop_flag["stop"] = True
            break

        elapsed = time.time() - start_tick
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
