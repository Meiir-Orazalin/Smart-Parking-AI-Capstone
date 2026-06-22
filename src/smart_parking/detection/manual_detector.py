"""
Smart Parking AI: Improved Detection Script
ISTE-500/501 Senior Development Project
Team: Dilshod, Walid, Sayed, Meiir

This improved version has two modes:
1. VEHICLE DETECTION MODE - Detects all vehicles (great for demos!)
2. SLOT DEFINITION MODE - Click to define parking spaces

Usage:
    python smart_parking_v2.py image.jpg              # Vehicle detection
    python smart_parking_v2.py image.jpg --define     # Define slots interactively
    python smart_parking_v2.py image.jpg --config slots.json  # Use saved slots
"""

import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from smart_parking.slots import (
    SeedValidationError,
    default_preview_path,
    generate_slots_from_seed_file,
    load_anchor_seed_file,
    render_generated_slots_preview,
    save_anchor_seed_file,
)
from smart_parking.utils.paths import default_vehicle_model_path


class SmartParkingV2:
    """Improved Smart Parking Detection with multiple modes."""
    
    def __init__(self, model_path='yolov8m.pt'):
        """Initialize with YOLO model."""
        print("🚗 Initializing Smart Parking AI v2...")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            fallback_model = default_vehicle_model_path()
            if fallback_model.exists():
                resolved_model_path = fallback_model
        self.model = YOLO(str(resolved_model_path))
        
        # Vehicle classes in COCO: car, motorcycle, bus, truck
        self.vehicle_classes = [2, 3, 5, 7]
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # For interactive slot definition
        self.slots = []
        self.current_points = []
        self.defining_slots = False
        
        print("✅ Model loaded!")
    
    def detect_vehicles(self, image_path, output_path=None, show=True, conf_threshold=0.3):
        """
        MODE 1: Detect all vehicles in the image.
        This is the best mode for demos - clearly shows AI detection working.
        """
        print(f"\n📸 Detecting vehicles in: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = frame.shape[:2]
        
        # Run YOLO detection
        print("  🤖 Running YOLOv8 detection...")
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        # Process detections
        vehicles = []
        output = frame.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                
                if cls in self.vehicle_classes:
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': self.class_names.get(cls, 'vehicle')
                    })
                    
                    # Draw bounding box
                    color = (0, 0, 255)  # Red for vehicles
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with confidence
                    label = f"{self.class_names.get(cls, 'vehicle')} {conf*100:.1f}%"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(output, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(output, label, (x1, y1 - 7), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw stats panel
        self._draw_vehicle_stats(output, vehicles)
        
        # Print results
        print(f"\n  ✅ Detection Complete!")
        print(f"     - Vehicles Found: {len(vehicles)}")
        for v in vehicles:
            print(f"       • {v['class']}: {v['confidence']*100:.1f}%")
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, output)
            print(f"     - Saved to: {output_path}")
        
        # Display
        if show:
            # Resize for display if too large
            display = self._resize_for_display(output)
            cv2.imshow('Smart Parking AI - Vehicle Detection', display)
            print("\n  Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return output, vehicles
    
    def detect_with_slots(self, image_path, slots_config, output_path=None, show=True, conf_threshold=0.2, overlap_threshold=0.2, imgsz=1280, all_classes=False, pad_ratio=0.1, unsure_threshold=0.8):
        """
        MODE 2: Detect vehicles and check slot occupancy using predefined slots.
        """
        print(f"\n📸 Processing with slot configuration...")
        
        # Load slots from config
        slots = self._load_slots(slots_config)
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run YOLO detection
        results = self.model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)
        
        # Get vehicle bounding boxes
        vehicle_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if all_classes or cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_boxes.append((x1, y1, x2, y2))

        # If no vehicles detected by class, fall back to size/aspect filtering
        if not vehicle_boxes and len(results) > 0 and results[0].boxes is not None:
            h, w = frame.shape[:2]
            img_area = w * h
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                area = bw * bh
                aspect = bw / bh if bh > 0 else 0
                # Heuristic: cars are moderate size, not tiny or huge
                if area < img_area / 5000 or area > img_area / 6:
                    continue
                if aspect < 0.5 or aspect > 3.5:
                    continue
                vehicle_boxes.append((x1, y1, x2, y2))

        print(f"  🚗 Vehicles detected: {len(vehicle_boxes)}")
        
        # Check each slot for occupancy
        output = frame.copy()
        occupied_count = 0
        
        uncertain_count = 0
        for i, slot in enumerate(slots):
            # Get slot polygon points
            points = np.array(slot['points'], dtype=np.int32)
            
            # Check if any vehicle overlaps with this slot
            is_occupied = False
            slot_confidence = 0
            
            for vbox in vehicle_boxes:
                vx1, vy1, vx2, vy2 = vbox
                # Pad vehicle box to be more tolerant to slight misalignment
                pad_x = int((vx2 - vx1) * pad_ratio)
                pad_y = int((vy2 - vy1) * pad_ratio)
                pvx1 = max(0, vx1 - pad_x)
                pvy1 = max(0, vy1 - pad_y)
                pvx2 = min(frame.shape[1], vx2 + pad_x)
                pvy2 = min(frame.shape[0], vy2 + pad_y)
                overlap = self._check_overlap(points, (pvx1, pvy1, pvx2, pvy2))
                if overlap > overlap_threshold:  # overlap threshold
                    is_occupied = True
                    slot_confidence = overlap
                    break
                # Fallback: vehicle center inside polygon
                vcx = int((vx1 + vx2) / 2)
                vcy = int((vy1 + vy2) / 2)
                if cv2.pointPolygonTest(points, (vcx, vcy), False) >= 0:
                    is_occupied = True
                    slot_confidence = max(slot_confidence, 0.5)
                    break
            
            is_unsure = 0.0 < slot_confidence < unsure_threshold

            # Draw slot
            if is_unsure:
                color = (0, 255, 255)  # Yellow
                label = f"#{i+1} UNSURE"
                uncertain_count += 1
            elif is_occupied:
                color = (0, 0, 255)  # Red
                label = f"#{i+1} OCCUPIED"
                occupied_count += 1
            else:
                color = (0, 255, 0)  # Green
                label = f"#{i+1} AVAILABLE"
            
            # Draw polygon
            cv2.polylines(output, [points], True, color, 3)
            
            # Fill with transparency
            overlay = output.copy()
            cv2.fillPoly(overlay, [points], color)
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
            
            # Draw label
            centroid = points.mean(axis=0).astype(int)
            cv2.putText(output, label, (centroid[0] - 50, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if is_occupied and slot_confidence > 0:
                conf_text = f"{slot_confidence*100:.1f}%"
                cv2.putText(output, conf_text, (centroid[0] - 20, centroid[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw stats
        self._draw_slot_stats(output, len(slots), occupied_count, uncertain_count)
        
        # Save and show
        if output_path:
            cv2.imwrite(output_path, output)
        
        if show:
            display = self._resize_for_display(output)
            cv2.imshow('Smart Parking AI - Slot Detection', display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return output

    def detect_with_slots_frame(
        self,
        frame,
        slots,
        conf_threshold=0.2,
        overlap_threshold=0.2,
        imgsz=1280,
        all_classes=False,
        pad_ratio=0.1,
        unsure_threshold=0.8,
        dynamic_occupancy_threshold=False,
        near_occupancy_threshold=0.55,
        far_occupancy_threshold=0.30,
        dynamic_threshold_axis="y",
    ):
        """
        Same as detect_with_slots but operates on a frame already in memory.
        Returns (output_frame, slots_with_status, occupied_count, available_count, uncertain_count, vehicle_boxes, person_boxes)
        """
        if frame is None:
            raise ValueError("Frame is None")

        slots = self._load_slots(slots)

        # Run YOLO detection
        results = self.model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)

        # Get vehicle and person bounding boxes
        vehicle_boxes = []
        person_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls_id == 0:
                    person_boxes.append((x1, y1, x2, y2))
                if all_classes or cls_id in self.vehicle_classes:
                    vehicle_boxes.append((x1, y1, x2, y2))

        # If no vehicles detected by class, fall back to size/aspect filtering
        if not vehicle_boxes and len(results) > 0 and results[0].boxes is not None:
            h, w = frame.shape[:2]
            img_area = w * h
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                area = bw * bh
                aspect = bw / bh if bh > 0 else 0
                # Heuristic: cars are moderate size, not tiny or huge
                if area < img_area / 5000 or area > img_area / 6:
                    continue
                if aspect < 0.5 or aspect > 3.5:
                    continue
                vehicle_boxes.append((x1, y1, x2, y2))

        output = frame.copy()
        occupied_count = 0
        uncertain_count = 0
        slot_statuses = []

        for i, slot in enumerate(slots):
            points = np.array(slot['points'], dtype=np.int32)
            is_occupied = False
            slot_confidence = 0
            slot_occupancy_threshold = self._slot_dynamic_threshold(
                points,
                frame.shape[:2],
                base_threshold=unsure_threshold,
                enabled=dynamic_occupancy_threshold,
                near_threshold=near_occupancy_threshold,
                far_threshold=far_occupancy_threshold,
                axis=dynamic_threshold_axis,
            )
            slot_overlap_threshold = min(float(overlap_threshold), slot_occupancy_threshold)

            for vbox in vehicle_boxes:
                vx1, vy1, vx2, vy2 = vbox
                # Pad vehicle box to be more tolerant to slight misalignment
                pad_x = int((vx2 - vx1) * pad_ratio)
                pad_y = int((vy2 - vy1) * pad_ratio)
                pvx1 = max(0, vx1 - pad_x)
                pvy1 = max(0, vy1 - pad_y)
                pvx2 = min(frame.shape[1], vx2 + pad_x)
                pvy2 = min(frame.shape[0], vy2 + pad_y)
                overlap = self._check_overlap(points, (pvx1, pvy1, pvx2, pvy2))
                if overlap > slot_overlap_threshold:
                    is_occupied = True
                    slot_confidence = overlap
                    break

                vcx = int((vx1 + vx2) / 2)
                vcy = int((vy1 + vy2) / 2)
                if cv2.pointPolygonTest(points, (vcx, vcy), False) >= 0:
                    is_occupied = True
                    slot_confidence = max(slot_confidence, 0.5)
                    break

            is_unsure = 0.0 < slot_confidence < slot_occupancy_threshold

            if is_unsure:
                color = (0, 255, 255)
                label = f"#{i+1} UNSURE"
                uncertain_count += 1
            elif is_occupied:
                color = (0, 0, 255)
                label = f"#{i+1} OCCUPIED"
                occupied_count += 1
            else:
                color = (0, 255, 0)
                label = f"#{i+1} AVAILABLE"

            slot_statuses.append({
                "id": i,
                "occupied": is_occupied and not is_unsure,
                "unsure": is_unsure,
                "status": "unsure" if is_unsure else ("occupied" if is_occupied else "available"),
                "confidence": slot_confidence,
                "overlap_threshold": slot_overlap_threshold,
                "occupancy_threshold": slot_occupancy_threshold,
                "label": label,
                "points": slot["points"],
            })

            cv2.polylines(output, [points], True, color, 3)
            overlay = output.copy()
            cv2.fillPoly(overlay, [points], color)
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)

            centroid = points.mean(axis=0).astype(int)
            cv2.putText(output, label, (centroid[0] - 50, centroid[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if is_occupied and slot_confidence > 0:
                conf_text = f"{slot_confidence*100:.1f}%"
                cv2.putText(output, conf_text, (centroid[0] - 20, centroid[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        available_count = len(slots) - occupied_count - uncertain_count
        self._draw_slot_stats(output, len(slots), occupied_count, uncertain_count)
        return output, slot_statuses, occupied_count, available_count, uncertain_count, vehicle_boxes, person_boxes

    @staticmethod
    def _load_slots(slots_config):
        if isinstance(slots_config, str):
            with open(slots_config, 'r') as f:
                slots = json.load(f)
        else:
            slots = slots_config
        return slots
    
    def define_slots_interactive(self, image_path, output_config='slots.json', seed_mode=False):
        """
        MODE 3: Interactively define parking slots by clicking.
        
        Instructions:
        - Click 4 corners of each parking slot (clockwise or counter-clockwise)
        - Press 'n' to finish current slot and start next
        - Press 's' to save all slots
        - Press 'u' to undo last point
        - Press 'r' to reset current slot
        - Press 'q' to quit
        """
        print(f"\n🖱️  Interactive {'Anchor Seed' if seed_mode else 'Slot'} Definition Mode")
        print("=" * 50)
        print("Instructions:")
        print("  • Click 4 corners of each parking slot")
        print("  • Press 'n' = finish slot, start next")
        print("  • Press 's' = save all slots to file")
        print("  • Press 'u' = undo last point")
        print("  • Press 'r' = reset current slot")
        print("  • Press 'q' = quit")
        if seed_mode:
            print("  • After each anchor, enter: row slot_index slot_count")
        print("=" * 50)
        
        # Read image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.image = self.original_image.copy()
        self.slots = []
        self.current_points = []
        self.defining_slots = True
        
        # Create window and set mouse callback
        cv2.namedWindow('Define Parking Slots')
        cv2.setMouseCallback('Define Parking Slots', self._mouse_callback)
        
        while True:
            # Draw current state
            display = self.image.copy()
            
            # Draw completed slots
            for i, slot in enumerate(self.slots):
                points = np.array(slot['points'], dtype=np.int32)
                cv2.polylines(display, [points], True, (0, 255, 0), 2)
                centroid = points.mean(axis=0).astype(int)
                if seed_mode:
                    label = f"R{slot['row']} S{slot['slot_index']}/{slot['slot_count']}"
                    cv2.putText(display, label, (centroid[0] - 36, centroid[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    cv2.putText(display, f"#{i+1}", (centroid[0]-10, centroid[1]+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw current points being defined
            for point in self.current_points:
                cv2.circle(display, point, 5, (0, 0, 255), -1)
            
            if len(self.current_points) > 1:
                for i in range(len(self.current_points) - 1):
                    cv2.line(display, self.current_points[i], self.current_points[i+1], (0, 0, 255), 2)
            
            # Draw instructions
            cv2.putText(display, f"Slots defined: {len(self.slots)} | Current points: {len(self.current_points)}/4",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "n=next slot | s=save | u=undo | r=reset | q=quit",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Resize for display
            display = self._resize_for_display(display)
            cv2.imshow('Define Parking Slots', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):  # Next slot
                min_points = 4 if seed_mode else 3
                if len(self.current_points) >= min_points:
                    slot_payload = {'points': self.current_points.copy()}
                    if seed_mode:
                        slot_payload.update(self._prompt_anchor_metadata())
                        slot_payload["is_anchor"] = True
                    self.slots.append(slot_payload)
                    print(f"  ✓ Slot #{len(self.slots)} defined")
                    self.current_points = []
                else:
                    print(f"  ⚠ Need at least {min_points} points for this mode")
            
            elif key == ord('s'):  # Save
                if len(self.slots) > 0:
                    if seed_mode:
                        save_anchor_seed_file(
                            output_config,
                            self.slots,
                            image_shape=self.original_image.shape[:2],
                        )
                    else:
                        with open(output_config, 'w') as f:
                            json.dump(self.slots, f, indent=2)
                    print(f"\n  ✅ Saved {len(self.slots)} slots to {output_config}")
                else:
                    print("  ⚠ No slots to save")
            
            elif key == ord('u'):  # Undo
                if len(self.current_points) > 0:
                    self.current_points.pop()
                    print("  ↩ Undid last point")
            
            elif key == ord('r'):  # Reset current
                self.current_points = []
                print("  🔄 Reset current slot")
            
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        
        if len(self.slots) > 0:
            print(f"\n✅ Defined {len(self.slots)} parking {'anchors' if seed_mode else 'slots'}")
            print(f"   Config saved to: {output_config}")
            if seed_mode:
                print(
                    f"\n   To generate full slots: python smart_parking_v2.py {image_path} "
                    f"--generate-from-seeds {output_config} --output slots.json"
                )
            else:
                print(f"\n   To use: python smart_parking_v2.py {image_path} --config {output_config}")
        
        return self.slots

    @staticmethod
    def _prompt_anchor_metadata():
        """Prompt for row/slot metadata used by the interpolation workflow."""
        while True:
            raw = input("  Enter anchor metadata as: row slot_index slot_count > ").strip()
            parts = raw.replace(",", " ").split()
            if len(parts) != 3:
                print("  ⚠ Please enter exactly three integers, for example: 1 3 12")
                continue
            try:
                row, slot_index, slot_count = (int(part) for part in parts)
            except ValueError:
                print("  ⚠ Metadata must be integers")
                continue
            if row < 1 or slot_count < 2 or slot_index < 1 or slot_index > slot_count:
                print("  ⚠ Expected row >= 1 and 1 <= slot_index <= slot_count with slot_count >= 2")
                continue
            return {
                "row": row,
                "slot_index": slot_index,
                "slot_count": slot_count,
            }
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for slot definition."""
        if event == cv2.EVENT_LBUTTONDOWN and self.defining_slots:
            # Scale coordinates back if image was resized
            h, w = self.original_image.shape[:2]
            display_h, display_w = self._get_display_size(h, w)
            
            scale_x = w / display_w
            scale_y = h / display_h
            
            actual_x = int(x * scale_x)
            actual_y = int(y * scale_y)
            
            self.current_points.append((actual_x, actual_y))
            print(f"  📍 Point {len(self.current_points)}: ({actual_x}, {actual_y})")
            
            if len(self.current_points) == 4:
                print("  → 4 points reached. Press 'n' to confirm slot or 'r' to reset")
    
    def _check_overlap(self, polygon_points, bbox):
        """Check overlap between a polygon and a bounding box."""
        # Create masks
        x1, y1, x2, y2 = bbox
        
        # Get bounding rect of polygon
        px, py, pw, ph = cv2.boundingRect(polygon_points)
        
        # Quick check - if bounding boxes don't overlap, no need for detailed check
        if x2 < px or x1 > px + pw or y2 < py or y1 > py + ph:
            return 0
        
        # Create a small canvas for overlap calculation
        max_x = max(x2, px + pw) + 10
        max_y = max(y2, py + ph) + 10
        
        # Polygon mask
        poly_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [polygon_points], 255)
        
        # Bbox mask
        bbox_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Calculate intersection
        intersection = cv2.bitwise_and(poly_mask, bbox_mask)
        
        # Calculate areas
        poly_area = cv2.countNonZero(poly_mask)
        intersection_area = cv2.countNonZero(intersection)
        
        if poly_area == 0:
            return 0
        
        return intersection_area / poly_area

    @staticmethod
    def _slot_dynamic_threshold(
        points,
        frame_shape,
        *,
        base_threshold=0.2,
        enabled=False,
        near_threshold=0.55,
        far_threshold=0.30,
        axis="y",
    ):
        if not enabled:
            return float(base_threshold)

        if points is None or len(points) == 0 or not frame_shape:
            return float(base_threshold)

        height, width = frame_shape[:2]
        dimension = height if axis == "y" else width
        if dimension <= 1:
            return float(base_threshold)

        points_array = np.array(points, dtype=np.float32)
        coord_index = 1 if axis == "y" else 0
        position = float(points_array[:, coord_index].mean()) / float(dimension - 1)
        position = max(0.0, min(1.0, position))

        threshold = float(far_threshold) + position * (float(near_threshold) - float(far_threshold))
        return max(0.0, min(1.0, threshold))
    
    def _draw_vehicle_stats(self, frame, vehicles):
        """Draw statistics panel for vehicle detection mode."""
        height, width = frame.shape[:2]
        
        # Panel
        panel_width = 300
        panel_height = 180
        cv2.rectangle(frame, (width - panel_width - 10, 10), 
                     (width - 10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - panel_width - 10, 10), 
                     (width - 10, panel_height), (0, 200, 255), 2)
        
        # Title
        cv2.putText(frame, "SMART PARKING AI", 
                   (width - panel_width + 10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # Stats
        stats = [
            f"Vehicles Detected: {len(vehicles)}",
            "",
            "Breakdown:",
        ]
        
        # Count by type
        type_counts = {}
        for v in vehicles:
            vtype = v['class']
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        for vtype, count in type_counts.items():
            stats.append(f"  {vtype}: {count}")
        
        y = 70
        for stat in stats:
            cv2.putText(frame, stat, (width - panel_width + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 22
    
    def _draw_slot_stats(self, frame, total_slots, occupied, uncertain=0):
        """Draw statistics panel for slot detection mode."""
        height, width = frame.shape[:2]
        available = total_slots - occupied - uncertain
        
        # Panel
        panel_width = 220
        panel_height = 155
        panel_x = 10
        panel_y = 10
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (0, 200, 255), 2)
        
        # Title
        cv2.putText(frame, "SMART PARKING AI", 
                   (panel_x + 8, panel_y + 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        
        # Stats
        stats = [
            (f"Total Slots: {total_slots}", (255, 255, 255)),
            (f"Occupied: {occupied}", (0, 0, 255)),
            (f"Available: {available}", (0, 255, 0)),
            (f"Unsure: {uncertain}", (0, 255, 255)),
            (f"Occupancy: {occupied/total_slots*100:.1f}%", (255, 255, 255)),
        ]
        
        y = panel_y + 50
        for text, color in stats:
            cv2.putText(frame, text, (panel_x + 8, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 22
    
    def _get_display_size(self, height, width, max_dim=1200):
        """Calculate display size maintaining aspect ratio."""
        if max(height, width) <= max_dim:
            return height, width
        
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        
        return new_height, new_width
    
    def _resize_for_display(self, frame, max_dim=1200):
        """Resize frame for display if too large."""
        height, width = frame.shape[:2]
        new_height, new_width = self._get_display_size(height, width, max_dim)
        
        if (new_height, new_width) != (height, width):
            return cv2.resize(frame, (new_width, new_height))
        return frame


def main():
    """Main function with command line interface."""
    print("=" * 60)
    print("  🅿️  SMART PARKING AI v2")
    print("  Improved Detection with Multiple Modes")
    print("=" * 60)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python smart_parking_v2.py <image>              # Detect vehicles")
        print("  python smart_parking_v2.py <image> --define [--output slots.json]  # Define slots interactively")
        print("  python smart_parking_v2.py <image> --define-seeds [--output slot_anchors.json]  # Define row anchors")
        print("  python smart_parking_v2.py <image> --generate-from-seeds slot_anchors.json [--output slots.json]  # Fill remaining slots")
        print("  python smart_parking_v2.py <image> --config <slots.json> [--conf 0.2 --overlap 0.2]  # Use saved slots")
        print("\nExample:")
        print("  python smart_parking_v2.py parking_lot.jpg")
        print("  python smart_parking_v2.py parking_lot.jpg --define --output data/slots/parking_cropped_slots.json")
        print("  python smart_parking_v2.py parking_lot.jpg --define-seeds --output data/slots/parking_cropped_anchor_seeds.json")
        print("  python smart_parking_v2.py parking_lot.jpg --generate-from-seeds data/slots/parking_cropped_anchor_seeds.json --output data/slots/parking_cropped_slots.json")
        print("  python smart_parking_v2.py parking_lot.jpg --config my_slots.json")
        return
    
    image_path = sys.argv[1]
    output_config = 'slots.json'
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    # Determine mode
    if '--generate-from-seeds' in sys.argv:
        seed_idx = sys.argv.index('--generate-from-seeds') + 1
        if seed_idx >= len(sys.argv):
            print("❌ Error: Please provide a seed file path after --generate-from-seeds")
            return
        seed_file = sys.argv[seed_idx]
        if '--output' in sys.argv:
            output_idx = sys.argv.index('--output') + 1
            if output_idx < len(sys.argv):
                output_config = sys.argv[output_idx]
            else:
                print("❌ Error: Please provide a file path after --output")
                return
        else:
            output_config = "generated_slots.json"

        preview_path = None
        if '--preview' in sys.argv:
            preview_idx = sys.argv.index('--preview') + 1
            if preview_idx < len(sys.argv):
                preview_path = sys.argv[preview_idx]
            else:
                print("❌ Error: Please provide a file path after --preview")
                return

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Error: Could not read image: {image_path}")
            return

        try:
            slots = generate_slots_from_seed_file(seed_file, image_shape=frame.shape[:2])
            with open(output_config, 'w') as handle:
                json.dump(slots, handle, indent=2)

            seed_payload = load_anchor_seed_file(seed_file)
            anchors = seed_payload.get("anchors", [])
            preview_target = Path(preview_path) if preview_path else default_preview_path(output_config)
            preview = render_generated_slots_preview(frame, slots, anchors=anchors)
            cv2.imwrite(str(preview_target), preview)
        except SeedValidationError as exc:
            print(f"❌ Seed validation error: {exc}")
            return

        print(f"\n  ✅ Generated {len(slots)} slots from {seed_file}")
        print(f"     - Saved slots to: {output_config}")
        print(f"     - Saved preview to: {preview_target}")

    elif '--define-seeds' in sys.argv:
        if '--output' in sys.argv:
            output_idx = sys.argv.index('--output') + 1
            if output_idx < len(sys.argv):
                output_config = sys.argv[output_idx]
            else:
                print("❌ Error: Please provide a file path after --output")
                return
        else:
            output_config = 'slot_anchors.json'
        detector = SmartParkingV2()
        detector.define_slots_interactive(image_path, output_config=output_config, seed_mode=True)

    elif '--define' in sys.argv:
        if '--output' in sys.argv:
            output_idx = sys.argv.index('--output') + 1
            if output_idx < len(sys.argv):
                output_config = sys.argv[output_idx]
            else:
                print("❌ Error: Please provide a file path after --output")
                return
        detector = SmartParkingV2()
        detector.define_slots_interactive(image_path, output_config=output_config)
    
    elif '--config' in sys.argv:
        # Use predefined slots
        config_idx = sys.argv.index('--config') + 1
        if config_idx < len(sys.argv):
            config_file = sys.argv[config_idx]
            detector = SmartParkingV2()
            conf = 0.2
            overlap = 0.2
            imgsz = 1280
            all_classes = False
            pad_ratio = 0.1
            unsure = 0.8
            if '--conf' in sys.argv:
                try:
                    conf = float(sys.argv[sys.argv.index('--conf') + 1])
                except Exception:
                    pass
            if '--overlap' in sys.argv:
                try:
                    overlap = float(sys.argv[sys.argv.index('--overlap') + 1])
                except Exception:
                    pass
            if '--imgsz' in sys.argv:
                try:
                    imgsz = int(sys.argv[sys.argv.index('--imgsz') + 1])
                except Exception:
                    pass
            if '--all-classes' in sys.argv:
                all_classes = True
            if '--pad' in sys.argv:
                try:
                    pad_ratio = float(sys.argv[sys.argv.index('--pad') + 1])
                except Exception:
                    pass
            if '--unsure' in sys.argv:
                try:
                    unsure = float(sys.argv[sys.argv.index('--unsure') + 1])
                except Exception:
                    pass
            detector.detect_with_slots(
                image_path, 
                config_file,
                output_path='detection_result.jpg',
                conf_threshold=conf,
                overlap_threshold=overlap,
                imgsz=imgsz,
                all_classes=all_classes,
                pad_ratio=pad_ratio,
                unsure_threshold=unsure,
            )
        else:
            print("❌ Error: Please provide config file path after --config")
    
    else:
        # Default: Vehicle detection mode
        detector = SmartParkingV2()
        detector.detect_vehicles(
            image_path,
            output_path='detection_result.jpg',
            show=True
        )
    
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
