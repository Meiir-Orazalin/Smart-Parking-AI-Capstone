import os
import time

import cv2
from ultralytics import YOLO


def main():
    video_path = os.getenv("VIDEO_PATH", "CarPark.mov")
    model_path = os.getenv("YOLO_MODEL_PATH", "yolov8x.pt")
    conf = float(os.getenv("YOLO_CONF", "0.25"))
    iou = float(os.getenv("YOLO_IOU", "0.45"))
    scale = float(os.getenv("SCALE", "1.0"))
    frame_skip = int(os.getenv("FRAME_SKIP", "0"))
    max_fps = float(os.getenv("MAX_FPS", "20"))
    show_labels = os.getenv("SHOW_LABELS", "1") == "1"
    loop_video = os.getenv("LOOP_VIDEO", "1") == "1"
    min_area_frac = float(os.getenv("MIN_BOX_AREA_FRAC", "0.003"))
    max_area_frac = float(os.getenv("MAX_BOX_AREA_FRAC", "0.08"))

    # COCO vehicle classes: car=2, motorcycle=3, bus=5, truck=7
    # Set YOLO_CLASSES="" to disable class filtering.
    classes_env = os.getenv("YOLO_CLASSES", "2,3,5,7")
    classes = [int(c.strip()) for c in classes_env.split(",") if c.strip() != ""]
    if not classes:
        classes = None

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    override_fps = os.getenv("REALTIME_FPS")
    source_fps = float(override_fps) if override_fps else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_delay = 1.0 / source_fps if source_fps > 0 else 1.0 / 30.0
    delay_ms = max(1, int(frame_delay * 1000))
    print(f"Playback FPS target: {source_fps:.2f} (delay {delay_ms} ms), frame_skip={frame_skip}")

    frame_idx = 0
    last_time = 0.0

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

        now = time.time()
        if max_fps > 0 and now - last_time < (1.0 / max_fps):
            continue
        last_time = now

        if scale != 1.0:
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            frame_small = frame

        results = model(
            frame_small,
            conf=conf,
            iou=iou,
            classes=classes,
            verbose=False,
        )[0]

        out = frame_small.copy()
        img_h, img_w = out.shape[:2]
        min_area = min_area_frac * img_w * img_h
        max_area = max_area_frac * img_w * img_h
        if results.boxes is not None:
            for box, cls_id, score in zip(
                results.boxes.xyxy.tolist(),
                results.boxes.cls.tolist(),
                results.boxes.conf.tolist(),
            ):
                x1, y1, x2, y2 = [int(v) for v in box]
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < min_area or area > max_area:
                    continue
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if show_labels:
                    label = f"{int(cls_id)} {score:.2f}"
                    cv2.putText(
                        out,
                        label,
                        (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )

        cv2.imshow("YOLOv8 Vehicle Detections", out)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

        elapsed = time.time() - start_tick
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
