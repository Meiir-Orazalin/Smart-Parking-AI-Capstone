import json
import os
import threading
import time
from datetime import datetime, timezone

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from smart_parking_detection import SmartParkingV2


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


app = FastAPI(title="SmartPark Video Backend API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


state_lock = threading.Lock()
state = {
    "video_path": "",
    "has_frame": False,
    "frame_count": 0,
    "fps_estimate": 0.0,
    "last_frame_time": None,
    "last_infer_time": None,
    "occupied": 0,
    "available": 0,
    "unsure": 0,
    "capacity": 0,
}

camera_store = []

def load_slots(slots_path: str):
    if not os.path.exists(slots_path):
        return []
    with open(slots_path, "r") as f:
        return json.load(f)


def scale_slots(slots, scale_x: float, scale_y: float):
    scaled = []
    for s in slots:
        pts = []
        for (x, y) in s.get("points", []):
            pts.append([int(x * scale_x), int(y * scale_y)])
        if len(pts) >= 3:
            scaled.append({"points": pts})
    return scaled


def get_scale_from_source_image(source_image: str, frame_w: int, frame_h: int):
    if not source_image or not os.path.exists(source_image):
        return 1.0, 1.0
    try:
        from PIL import Image

        img = Image.open(source_image)
        src_w, src_h = img.size
        if src_w > 0 and src_h > 0:
            return frame_w / src_w, frame_h / src_h
    except Exception:
        pass
    return 1.0, 1.0


def inference_loop():
    video_path = os.getenv("VIDEO_PATH", "ParkingVideo.mov")
    slots_path = os.getenv("SLOTS_PATH", "slots.json")
    slots_source_image = os.getenv("SLOTS_SOURCE_IMAGE", "")

    conf = float(os.getenv("CONF_THRESHOLD", "0.01"))
    overlap = float(os.getenv("OVERLAP_THRESHOLD", "0.26"))
    imgsz = int(os.getenv("IMGSZ", "480"))
    pad = float(os.getenv("PAD_RATIO", "0.03"))
    unsure = float(os.getenv("UNSURE_THRESHOLD", "0.55"))
    smooth_frames = int(os.getenv("SMOOTH_FRAMES", "5"))
    frame_skip = int(os.getenv("FRAME_SKIP", "12"))
    display_scale = float(os.getenv("DISPLAY_SCALE", "0.6"))

    detector = SmartParkingV2()
    base_slots = load_slots(slots_path)
    capacity = len(base_slots)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with state_lock:
            state["video_path"] = video_path
        return

    with state_lock:
        state["video_path"] = video_path
        state["capacity"] = capacity

    states = [{"state": None, "streak": 0} for _ in range(capacity)]
    frame_idx = 0
    last_fps_time = time.time()
    last_fps_count = 0
    scaled_slots_cache = None
    scaled_dims = None

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            time.sleep(0.1)
            continue

        frame_idx += 1
        now = time.time()

        # Update fps estimate
        last_fps_count += 1
        if now - last_fps_time >= 1.0:
            with state_lock:
                state["fps_estimate"] = last_fps_count / max(0.001, now - last_fps_time)
            last_fps_time = now
            last_fps_count = 0

        # Optional downscale for speed
        orig_h, orig_w = frame.shape[:2]
        if display_scale != 1.0:
            frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

        h, w = frame.shape[:2]
        dims = (w, h, display_scale)
        if scaled_slots_cache is None or scaled_dims != dims:
            sx, sy = get_scale_from_source_image(slots_source_image, w, h)
            if display_scale != 1.0 and not slots_source_image:
                sx, sy = (w / orig_w) if orig_w else 1.0, (h / orig_h) if orig_h else 1.0
            scaled_slots_cache = scale_slots(base_slots, sx, sy)
            scaled_dims = dims

        process_this_frame = frame_skip <= 0 or (frame_idx % (frame_skip + 1)) == 1
        if process_this_frame and scaled_slots_cache:
            _, slot_list, occupied, available, uncertain = detector.detect_with_slots_frame(
                frame,
                scaled_slots_cache,
                conf_threshold=conf,
                overlap_threshold=overlap,
                imgsz=imgsz,
                pad_ratio=pad,
                unsure_threshold=unsure,
            )

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

            with state_lock:
                state["has_frame"] = True
                state["frame_count"] += 1
                state["last_frame_time"] = now
                state["last_infer_time"] = now
                state["occupied"] = int(occupied)
                state["available"] = int(available)
                state["unsure"] = int(uncertain)
                state["capacity"] = len(scaled_slots_cache)
        else:
            with state_lock:
                state["has_frame"] = True
                state["frame_count"] += 1
                state["last_frame_time"] = now


@app.on_event("startup")
def on_startup():
    if os.getenv("BACKEND_NO_LOOP") == "1":
        return
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()


def update_state(
    *,
    frame_time: float,
    occupied: int,
    available: int,
    unsure: int,
    capacity: int,
    fps_estimate: float | None = None,
):
    with state_lock:
        state["has_frame"] = True
        state["frame_count"] += 1
        state["last_frame_time"] = frame_time
        state["last_infer_time"] = frame_time
        state["occupied"] = int(occupied)
        state["available"] = int(available)
        state["unsure"] = int(unsure)
        state["capacity"] = int(capacity)
        if fps_estimate is not None:
            state["fps_estimate"] = float(fps_estimate)


@app.get("/status")
def get_status():
    with state_lock:
        last_frame_time = state["last_frame_time"]
        last_frame_age = None
        if last_frame_time is not None:
            last_frame_age = max(0.0, time.time() - last_frame_time)
        return {
            "status": {
                "video_path": state["video_path"],
                "has_frame": state["has_frame"],
                "frame_count": state["frame_count"],
                "fps_estimate": round(state["fps_estimate"], 2),
                "last_frame_age_seconds": last_frame_age,
                "last_infer_time": state["last_infer_time"],
            }
        }


@app.get("/lots")
def get_lots():
    with state_lock:
        capacity = int(state["capacity"] or 0)
        occupied = int(state["occupied"] or 0)
        available = int(state["available"] or 0)
        unsure = int(state["unsure"] or 0)
    occupancy_pct = (occupied / capacity * 100) if capacity > 0 else 0.0
    if occupancy_pct >= 90:
        status = "full"
    elif occupancy_pct >= 70:
        status = "almost_full"
    else:
        status = "available"
    return {
        "lots": [
            {
                "id": "main-lot",
                "name": "Main Lot",
                "occupied": occupied,
                "available": available,
                "unsure": unsure,
                "capacity": capacity,
                "distanceMi": 0.3,
                "status": status,
                "lastUpdated": iso_now(),
            }
        ]
    }


@app.get("/dashboard")
def get_dashboard():
    with state_lock:
        capacity = int(state["capacity"] or 0)
        occupied = int(state["occupied"] or 0)
        fps = float(state["fps_estimate"] or 0.0)
    occupancy_pct = int(round((occupied / capacity * 100) if capacity > 0 else 0.0))
    return {
        "dashboard": {
            "totalLots": 1,
            "occupancyPct": occupancy_pct,
            "activeAlerts": 0,
            "systemFps": round(fps, 2),
        }
    }


@app.get("/alerts")
def get_alerts():
    return {
        "alerts": [
            {
                "id": "alert-01",
                "title": "Camera Online",
                "severity": "low",
                "location": "Main Lot",
                "timeAgo": "Just now",
            }
        ]
    }


@app.get("/cameras")
def get_cameras():
    with state_lock:
        fps = float(state["fps_estimate"] or 0.0)
        stream_url = state["video_path"]
    base = {
        "id": "cam-01",
        "name": "Main Lot Camera",
        "fps": round(fps, 2),
        "uptime": "99.2%",
        "lastDetection": "Just now",
        "status": "online",
        "streamUrl": stream_url,
    }
    return {"cameras": [base] + camera_store}


@app.post("/cameras")
def create_camera(payload: dict):
    name = payload.get("name")
    location = payload.get("location")
    stream_url = payload.get("streamUrl")
    if not name or not location or not stream_url:
        raise HTTPException(status_code=400, detail="name, location, streamUrl required")
    if not stream_url.startswith("rtsp://") and not stream_url.startswith("http"):
        raise HTTPException(status_code=400, detail="streamUrl must start with rtsp:// or http")
    cam_id = f"cam-{len(camera_store) + 2}"
    cam = {
        "id": cam_id,
        "name": f"{name} - {location}",
        "fps": 0,
        "uptime": "100.0%",
        "lastDetection": "Just now",
        "status": "online",
        "streamUrl": stream_url,
    }
    camera_store.append(cam)
    return {"camera": cam}


@app.get("/profile")
def get_profile():
    return {
        "profile": {
            "name": "SmartPark Demo",
            "email": "demo@smartpark.local",
            "notificationsEnabled": True,
        }
    }


@app.get("/health")
def get_health():
    return {"status": "online", "time": iso_now()}
