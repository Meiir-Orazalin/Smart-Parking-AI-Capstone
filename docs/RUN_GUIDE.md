# Smart Parking Demo Run Guide

This guide explains how to run the project in its current form.

## 1. Requirements

- macOS or Linux
- Python virtual environment already created at `./venv`
- Project dependencies installed in that virtual environment
- A working video source:
  - local video file, or
  - MJPEG/HTTP/RTSP stream URL

This project is typically run from the repo root:

```bash
cd /Users/dilshod/Desktop/Repos/smart-parking-demo
```

## 2. Main Entry Point

The supported runtime entry point is:

```bash
./venv/bin/python run_demo.py
```

Compatibility wrappers also exist:

- `run_backend_video_with_ui.py`
- `backend_api_video.py`

Use `run_demo.py` unless you specifically need one of the wrappers.

## 3. Common Run Modes

### Manual mode with an existing slot file

```bash
./venv/bin/python run_demo.py \
  --slot-mode manual \
  --slots data/slots/slots.json
```

### Manual mode using anchor seeds

This is the preferred path when using the newer anchor-based slot workflow.

```bash
./venv/bin/python run_demo.py \
  --slot-mode manual \
  --anchor-seeds data/slots/parking_cropped_anchor_seeds.json
```

### Auto mode

```bash
./venv/bin/python run_demo.py \
  --slot-mode auto \
  --slots-cache data/slots/auto_slots.json \
  --slot-model assets/models/parking_slot_seg.pt
```

If `assets/models/parking_slot_seg.pt` is missing, auto mode can still reuse an existing cache and may fall back to manual slots.

## 4. Run With the Live MJPEG Stream

For the live stream used in the current demo:

```bash
./venv/bin/python run_demo.py \
  --video 'https://formal-usually-hay-yeah.trycloudflare.com/mjpeg' \
  --slot-mode manual \
  --anchor-seeds data/slots/parking_cropped_anchor_seeds.json \
  --api-host 127.0.0.1 \
  --api-port 8000
```

The app will:

- open the video stream
- generate the slot layout from the anchor seed file
- show the annotated OpenCV window
- start the FastAPI backend on `http://127.0.0.1:8000`

## 5. Run With Dynamic Thresholds

Dynamic thresholding adjusts the occupied/unsure threshold by slot position in the image. This helps with angled camera views where far and near slots behave differently.

```bash
./venv/bin/python run_demo.py \
  --video 'https://formal-usually-hay-yeah.trycloudflare.com/mjpeg' \
  --slot-mode manual \
  --anchor-seeds data/slots/parking_cropped_anchor_seeds.json \
  --dynamic-threshold \
  --far-threshold 0.30 \
  --near-threshold 0.55 \
  --show-thresholds \
  --api-host 127.0.0.1 \
  --api-port 8000
```

Useful tuning notes:

- lower `--far-threshold` if distant cars are missed
- raise `--far-threshold` if distant slots are falsely marked occupied
- raise `--near-threshold` if close slots are too easily marked occupied
- `--dynamic-axis y` is the default and is usually correct for top-to-bottom camera perspective

## 6. Define New Anchor Slots

The anchor editor is interactive and expects CLI input after each anchor is drawn.

First, make sure you have a still reference image. The current project uses:

```text
data/slots/live_anchor_reference.jpg
```

Then run:

```bash
./venv/bin/python smart_parking_detection.py \
  data/slots/live_anchor_reference.jpg \
  --define-seeds \
  --output data/slots/parking_cropped_anchor_seeds.json
```

Inside the OpenCV window:

- click 4 corners for one anchor slot
- press `n` to finish that anchor
- enter metadata in the terminal as:

```text
row slot_index slot_count
```

Example:

```text
1 1 12
1 12 12
2 1 12
2 12 12
```

Controls:

- `n` = finish current anchor
- `s` = save anchors
- `u` = undo last point
- `r` = reset current anchor
- `q` = quit

## 7. Generate Full Slots From Anchors

After defining anchors, generate the full slot layout and a preview image:

```bash
./venv/bin/python smart_parking_detection.py \
  data/slots/live_anchor_reference.jpg \
  --generate-from-seeds data/slots/parking_cropped_anchor_seeds.json \
  --output data/slots/parking_cropped_slots.json \
  --preview data/slots/parking_cropped_slots_preview.jpg
```

This produces:

- `data/slots/parking_cropped_slots.json`
- `data/slots/parking_cropped_slots_preview.jpg`

## 8. Verify the Backend

Once the app is running, verify the backend with:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/lots
curl http://127.0.0.1:8000/cameras
```

Important endpoints:

- `/health`
- `/status`
- `/lots`
- `/dashboard`
- `/cameras`

## 9. Useful Runtime Flags

- `--no-anonymize`
- `--anonymize-refresh-frames 3`
- `--frame-skip 12`
- `--display-scale 0.6`
- `--conf 0.01`
- `--overlap 0.26`
- `--unsure 0.55`
- `--smooth 5`

## 10. Stop the App

If the demo is running in the current terminal, press:

```text
q
```

inside the OpenCV window, or use `Ctrl+C` in the terminal.

If it was started in another terminal and you know the process ID:

```bash
kill <PID>
```

To find it:

```bash
pgrep -fl 'run_demo.py|smart_parking_detection.py|uvicorn|smart_parking'
```

## 11. Current Recommended Demo Command

For the current working setup, use:

```bash
./venv/bin/python run_demo.py \
  --video 'https://formal-usually-hay-yeah.trycloudflare.com/mjpeg' \
  --slot-mode manual \
  --anchor-seeds data/slots/parking_cropped_anchor_seeds.json \
  --dynamic-threshold \
  --far-threshold 0.30 \
  --near-threshold 0.55 \
  --show-thresholds \
  --api-host 127.0.0.1 \
  --api-port 8000
```
