import json
import os
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


app = FastAPI(title="SmartPark Backend API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


state_lock = threading.Lock()
state = {
    "rtsp_url": "",
    "has_frame": False,
    "frame_count": 0,
    "fps_estimate": 0.0,
    "last_frame_time": None,
    "last_infer_time": None,
    "occupied": 0,
    "capacity": 0,
}

camera_store = []


def load_slots_scaled(slots_path: str, slots_source_image: str, frame_w: int, frame_h: int, scale: float):
    if not os.path.exists(slots_path):
        return []

    slots_scale_x = 1.0
    slots_scale_y = 1.0
    if slots_source_image and os.path.exists(slots_source_image):
        try:
            from PIL import Image

            img = Image.open(slots_source_image)
            src_w, src_h = img.size
            if src_w > 0 and src_h > 0 and frame_w > 0 and frame_h > 0:
                slots_scale_x = frame_w / src_w
                slots_scale_y = frame_h / src_h
        except Exception:
            pass

    slots_scaled = []
    with open(slots_path, "r") as f:
        slots = json.load(f)
    for s in slots:
        pts = s.get("points", [])
        if len(pts) < 3:
            continue
        pts_scaled = np.array(
            [[p[0] * slots_scale_x * scale, p[1] * slots_scale_y * scale] for p in pts],
            dtype=np.float32,
        )
        hull = cv2.convexHull(pts_scaled)
        area = float(cv2.contourArea(hull))
        if area <= 1.0:
            continue
        slots_scaled.append({"points": hull, "area": area})
    return slots_scaled


def compute_occupied(slots_scaled, preds, occ_min_conf: float, slot_occ_ioa: float):
    occ_boxes = []
    for p in preds:
        label = p.get("class", "")
        conf = float(p.get("confidence", 0.0))
        if label.lower() != "occupied" or conf < occ_min_conf:
            continue
        x = int(p["x"])
        y = int(p["y"])
        w = int(p["width"])
        h = int(p["height"])
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        occ_boxes.append((x1, y1, x2, y2))

    occupied_count = 0
    for slot in slots_scaled:
        pts = slot["points"]
        slot_area = slot["area"]
        is_occ = False
        for (x1, y1, x2, y2) in occ_boxes:
            rect = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )
            inter_area, _ = cv2.intersectConvexConvex(pts, rect)
            if slot_area > 0 and (inter_area / slot_area) >= slot_occ_ioa:
                is_occ = True
                break
        if is_occ:
            occupied_count += 1
    return occupied_count


def inference_loop():
    video_path = os.getenv(
        "VIDEO_PATH",
        "https://requested-illustrations-puzzle-emma.trycloudflare.com/mjpeg",
    )
    slots_path = os.getenv("ROI_SLOTS_PATH", "slots.json")
    slots_source_image = os.getenv("SLOTS_SOURCE_IMAGE")

    scale = float(os.getenv("SCALE", "1.0"))
    jpeg_quality = int(os.getenv("JPEG_QUALITY", "90"))
    infer_every_sec = float(os.getenv("INFER_EVERY_SEC", "2.0"))
    occ_min_conf = float(os.getenv("OCC_MIN_CONF", "0.3"))
    slot_occ_ioa = float(os.getenv("SLOT_OCCUPIED_IOA", "0.2"))

    api_key = os.getenv("ROBOFLOW_API_KEY")
    model_id = os.getenv("ROBOFLOW_MODEL_ID", "parking-spaces-ezhxz/1")
    client = None
    if api_key:
        client = InferenceHTTPClient.init(
            api_url="https://detect.roboflow.com",
            api_key=api_key,
        )

    with state_lock:
        state["rtsp_url"] = video_path

    cap = None
    slots_scaled = []
    last_infer = 0.0
    fps_ema = 0.0
    alpha = 0.1

    prev_frame_time = None
    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            time.sleep(0.2)
            if not cap.isOpened():
                time.sleep(1.0)
                continue

        ok, frame = cap.read()
        now = time.time()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        if not slots_scaled:
            slots_scaled = load_slots_scaled(slots_path, slots_source_image, w, h, scale)
            with state_lock:
                state["capacity"] = len(slots_scaled)

        with state_lock:
            state["has_frame"] = True
            state["frame_count"] += 1
            state["last_frame_time"] = now

        if prev_frame_time is not None:
            dt = now - prev_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
                with state_lock:
                    state["fps_estimate"] = fps_ema
        prev_frame_time = now

        if now - last_infer < infer_every_sec:
            continue

        if client is None:
            last_infer = now
            with state_lock:
                state["last_infer_time"] = now
            continue

        frame_small = (
            cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        )
        tmp_path = "rf_frame.jpg"
        cv2.imwrite(tmp_path, frame_small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        try:
            result = client.infer(tmp_path, model_id=model_id)
            preds = result.get("predictions", [])
            occupied = compute_occupied(slots_scaled, preds, occ_min_conf, slot_occ_ioa)
            with state_lock:
                state["occupied"] = occupied
                state["last_infer_time"] = now
        except Exception:
            with state_lock:
                state["last_infer_time"] = now
        last_infer = now


@app.on_event("startup")
def on_startup():
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()


@app.get("/status")
def get_status():
    with state_lock:
        last_frame_time = state["last_frame_time"]
        last_frame_age = None
        if last_frame_time is not None:
            last_frame_age = max(0.0, time.time() - last_frame_time)
        return {
            "status": {
                "rtsp_url": state["rtsp_url"],
                "has_frame": state["has_frame"],
                "frame_count": state["frame_count"],
                "fps_estimate": round(state["fps_estimate"], 2),
                "last_frame_age_seconds": last_frame_age,
            }
        }


@app.get("/lots")
def get_lots():
    with state_lock:
        capacity = int(state["capacity"] or 0)
        occupied = int(state["occupied"] or 0)
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
        stream_url = state["rtsp_url"]
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
