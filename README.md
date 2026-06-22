# Smart Parking Demo

This repo now has one supported runtime path and archives the old experiments separately.

This repository is a portfolio mirror of the RIT Dubai Smart Parking AI capstone. It combines the team's computer-vision/backend prototype with Meiir Orazalin's Expo/React Native mobile client. See [CONTRIBUTORS.md](CONTRIBUTORS.md) for attribution and contribution context.

## Main Workflow

- `run_demo.py`
  - launches the packaged video demo under `src/smart_parking/`
- `run_backend_video_with_ui.py`
  - same runtime entrypoint, kept as a compatibility wrapper
- `backend_api_video.py`
  - compatibility wrapper for the packaged FastAPI backend

## Supported Modes

- `manual`
  - loads slot polygons from `data/slots/slots.json`
- `auto`
  - tries to match a cached preset CCTV view from `data/slots/auto_slots.json`
  - if no cached view matches and a slot segmentation model is available, calibrates slots for the new stable view and appends it to the cache
  - if auto calibration is unavailable or fails and manual slots exist, falls back to manual slots

## Repo Layout

- `src/smart_parking/`
  - packaged app, API, slot cache/view state, occupancy engine, and detector integrations
- `assets/videos/`
  - local sample videos (not tracked because of repository size limits)
- `assets/images/`
  - sample stills and source images
- `assets/models/`
  - local model weights (not tracked; Ultralytics can download standard YOLO weights)
- `data/slots/`
  - manual slots and auto slot cache files
- `docs/`
  - specs, notes, and historical documentation
- `archive/legacy/`
  - superseded runners and heuristic experiments
- `archive/generated/`
  - generated debug and result images
- `mobile-app/`
  - Expo/React Native client for driver and parking-operations workflows

## Commands

Manual slots:

```bash
./venv/bin/python run_demo.py --slot-mode manual
```

Anonymization is enabled by default and blurs detected faces and license plates in the live demo output. Disable it with `--no-anonymize` or tune the refresh cadence with `--anonymize-refresh-frames 3`.

Auto mode with cached views and optional learned slot model:

```bash
./venv/bin/python run_demo.py \
  --slot-mode auto \
  --slots-cache data/slots/auto_slots.json \
  --slot-model assets/models/parking_slot_seg.pt
```

If `assets/models/parking_slot_seg.pt` is not present, auto mode can still reuse an existing cache or fall back to manual slots when available.

Prepare the `ParkingCropped` starter dataset after you create `data/slots/parking_cropped_slots.json`:

```bash
./venv/bin/python prepare_slot_dataset.py
```

Train a slot-segmentation model:

```bash
./venv/bin/python train_slot_model.py train \
  --data-yaml data/training/parking_cropped_dataset/data.yaml \
  --model yolov8n-seg.pt \
  --epochs 100
```

## Notes

- The old classical CV auto-slot scripts were moved to `archive/legacy/` because they are not part of the supported runtime path anymore.
- The occupancy path still uses the existing `SmartParkingV2` detector logic, but it now runs behind a separate occupancy engine that accepts canonical slot polygons from either manual JSON or the auto cache.
- The FastAPI status payload now includes slot/view state fields such as `slot_mode`, `active_view_id`, `view_status`, `cache_ready`, `calibration_status`, and anonymization settings.
- The starter manifest now lives at `data/training/parking_cropped_manifest.json` and targets `ParkingCropped.jpeg`, `ParkingCropped2.jpeg`, and `ParkingCropped3.jpeg`.
- You still need to create `data/slots/parking_cropped_slots.json` by labeling `ParkingCropped.jpeg` once before dataset preparation.
- Detailed training steps are in `docs/TRAIN_SLOT_MODEL.md`.

## Mobile App

The mobile client supports:

- driver and operations modes
- parking-lot availability and occupancy details
- interactive busyness predictions
- camera and alert management
- mock, hybrid, and API-backed data modes
- typed backend contracts and local persistence

```bash
cd mobile-app
npm install
npm start
```

Copy `mobile-app/.env.example` to `mobile-app/.env` to configure the data mode and backend URL. Additional setup guidance is available in [docs/SMARTPARKAI_MOBILE_RUN_GUIDE.md](docs/SMARTPARKAI_MOBILE_RUN_GUIDE.md).

## Backend Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py --slot-mode manual
```

Place a test video in `assets/videos/` before running the demo. Large videos and pretrained `.pt` weights are intentionally excluded from Git; standard Ultralytics model weights are downloaded automatically when requested by model name.
