# Train The Slot Model

Use these images as the starter set:

- [ParkingCropped.jpeg](/Users/dilshod/Desktop/Repos/smart-parking-demo/assets/images/ParkingCropped.jpeg)
- [ParkingCropped2.jpeg](/Users/dilshod/Desktop/Repos/smart-parking-demo/assets/images/ParkingCropped2.jpeg)
- [ParkingCropped3.jpeg](/Users/dilshod/Desktop/Repos/smart-parking-demo/assets/images/ParkingCropped3.jpeg)

`ParkingCropped_slots.jpg` is intentionally not used.

## 1. Label One Reference Image

Create `data/slots/parking_cropped_slots.json` from [ParkingCropped.jpeg](/Users/dilshod/Desktop/Repos/smart-parking-demo/assets/images/ParkingCropped.jpeg):

```bash
./venv/bin/python smart_parking_detection.py assets/images/ParkingCropped.jpeg --define
```

Save the slots as:

```text
data/slots/parking_cropped_slots.json
```

The dataset prep step will scale that geometry onto `ParkingCropped2.jpeg` and `ParkingCropped3.jpeg`.

## 2. Build The YOLO-Seg Dataset

```bash
./venv/bin/python prepare_slot_dataset.py \
  --manifest data/training/parking_cropped_manifest.json \
  --output-dir data/training/parking_cropped_dataset
```

## 3. Train The Segmentation Model

```bash
./venv/bin/python train_slot_model.py train \
  --data-yaml data/training/parking_cropped_dataset/data.yaml \
  --model yolov8n-seg.pt \
  --epochs 100 \
  --imgsz 1024 \
  --project runs/train \
  --name parking_slot_segmentation
```

After training, copy the best checkpoint to:

```text
assets/models/parking_slot_seg.pt
```

## 4. Run Auto Mode

```bash
./venv/bin/python run_demo.py \
  --slot-mode auto \
  --slot-model assets/models/parking_slot_seg.pt \
  --slots-cache data/slots/auto_slots.json
```

## Extra Frames

If you need more data from the same camera:

```bash
./venv/bin/python extract_slot_frames.py \
  --video assets/videos/ParkingVideo.MOV \
  --output-dir data/training/extracted_frames \
  --every-n-frames 60
```
