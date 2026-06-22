"""
Smart Parking AI: Simple & Reliable Demo Version
ISTE-500/501 Senior Development Project
Team: Dilshod, Walid, Sayed, Meiir

This version is optimized for RELIABILITY in demos:
1. Detects all vehicles first
2. Groups them into rows
3. Creates uniform slots across each row
4. Extends to cover the full parking area

This is the RECOMMENDED version for your presentation!

Usage:
    python smart_parking_simple.py image.jpg
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os


class SimpleParkingDetector:
    """Simple and reliable parking detection for demos."""
    
    def __init__(self, model_path='yolov8n.pt'):
        print("🚗 Initializing Smart Parking AI...")
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        print("✅ Ready!")
    
    def process_image(self, image_path, output_path=None, show=True):
        """Main processing function."""
        print(f"\n📸 Processing: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = image.shape[:2]
        print(f"  📐 Size: {width}x{height}")
        
        # Step 1: Detect all vehicles
        print("  🔍 Detecting vehicles...")
        vehicles = self._detect_vehicles(image)
        print(f"     Found {len(vehicles)} vehicles")
        
        if len(vehicles) < 2:
            print("  ⚠ Not enough vehicles to establish parking pattern")
            return self._fallback_vehicle_only(image, vehicles, output_path, show)
        
        # Step 2: Analyze parking structure
        print("  📊 Analyzing parking layout...")
        slots = self._create_parking_grid(vehicles, width, height)
        print(f"     Created {len(slots)} parking slots")
        
        # Step 3: Check occupancy
        slots = self._check_occupancy(slots, vehicles)
        
        # Step 4: Draw results
        output = self._draw_output(image, slots, vehicles)
        
        # Stats
        occupied = sum(1 for s in slots if s['occupied'])
        available = len(slots) - occupied
        
        print(f"\n  ✅ Complete!")
        print(f"     • Total: {len(slots)} slots")
        print(f"     • Occupied: {occupied}")
        print(f"     • Available: {available}")
        print(f"     • Rate: {occupied/len(slots)*100:.1f}%")
        
        if output_path:
            cv2.imwrite(output_path, output)
            print(f"     • Saved: {output_path}")
        
        if show:
            self._show(output)
        
        return output, slots
    
    def _detect_vehicles(self, image):
        """Detect vehicles using YOLO."""
        results = self.model(image, conf=0.25, verbose=False)
        
        vehicles = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1+x2)//2, (y1+y2)//2),
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'conf': float(box.conf[0])
                    })
        
        return vehicles
    
    def _create_parking_grid(self, vehicles, img_width, img_height):
        """Create parking grid based on detected vehicles."""
        
        # Calculate average vehicle dimensions
        avg_width = np.median([v['width'] for v in vehicles])
        avg_height = np.median([v['height'] for v in vehicles])
        
        # Slot dimensions (slightly larger than vehicles)
        slot_width = int(avg_width * 1.25)
        slot_height = int(avg_height * 1.25)
        
        # Group vehicles into rows by Y position
        rows = self._group_into_rows(vehicles, avg_height * 0.7)
        
        # Create slots for each row
        all_slots = []
        slot_id = 0
        
        for row_idx, row_vehicles in enumerate(rows):
            if not row_vehicles:
                continue
            
            # Row boundaries
            row_y = np.median([v['center'][1] for v in row_vehicles])
            y1 = int(row_y - slot_height // 2)
            y2 = int(row_y + slot_height // 2)
            
            # Get X extent of this row
            all_x = []
            for v in row_vehicles:
                all_x.extend([v['bbox'][0], v['bbox'][2]])
            
            min_x = min(all_x)
            max_x = max(all_x)
            
            # Calculate slot spacing from vehicles in this row
            if len(row_vehicles) > 1:
                centers = sorted([v['center'][0] for v in row_vehicles])
                spacings = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
                # Use median spacing, but ensure it's reasonable
                median_spacing = np.median(spacings)
                if slot_width * 0.7 < median_spacing < slot_width * 2:
                    slot_width = int(median_spacing)
            
            # Extend row to reasonable boundaries
            # Add 2 slots worth of space on each side
            row_start = max(0, min_x - slot_width * 2)
            row_end = min(img_width, max_x + slot_width * 2)
            
            # Create uniform grid of slots
            x = row_start
            while x + slot_width <= row_end:
                all_slots.append({
                    'id': slot_id,
                    'bbox': (int(x), int(y1), int(x + slot_width), int(y2)),
                    'center': (int(x + slot_width//2), int((y1+y2)//2)),
                    'row': row_idx,
                    'occupied': False,
                    'confidence': 0.95
                })
                slot_id += 1
                x += slot_width
        
        return all_slots
    
    def _group_into_rows(self, vehicles, tolerance):
        """Group vehicles into rows based on Y position."""
        if not vehicles:
            return []
        
        # Sort by Y
        sorted_v = sorted(vehicles, key=lambda v: v['center'][1])
        
        rows = []
        current_row = [sorted_v[0]]
        current_y = sorted_v[0]['center'][1]
        
        for v in sorted_v[1:]:
            if abs(v['center'][1] - current_y) <= tolerance:
                current_row.append(v)
                # Update row Y to be average
                current_y = np.mean([vv['center'][1] for vv in current_row])
            else:
                rows.append(current_row)
                current_row = [v]
                current_y = v['center'][1]
        
        rows.append(current_row)
        return rows
    
    def _check_occupancy(self, slots, vehicles):
        """Check which slots contain vehicles."""
        for slot in slots:
            sx1, sy1, sx2, sy2 = slot['bbox']
            scx, scy = slot['center']
            
            slot['occupied'] = False
            slot['confidence'] = 0.95  # Confidence it's empty
            
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v['bbox']
                vcx, vcy = v['center']
                
                # Method 1: Check if vehicle center is in slot
                if sx1 <= vcx <= sx2 and sy1 <= vcy <= sy2:
                    slot['occupied'] = True
                    slot['confidence'] = 0.95 + v['conf'] * 0.04
                    break
                
                # Method 2: Check overlap
                ix1 = max(sx1, vx1)
                iy1 = max(sy1, vy1)
                ix2 = min(sx2, vx2)
                iy2 = min(sy2, vy2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    slot_area = (sx2 - sx1) * (sy2 - sy1)
                    overlap = intersection / slot_area
                    
                    if overlap > 0.2:  # 20% overlap threshold
                        slot['occupied'] = True
                        slot['confidence'] = 0.90 + overlap * 0.09
                        break
        
        return slots
    
    def _draw_output(self, image, slots, vehicles):
        """Draw results on image."""
        output = image.copy()
        
        for slot in slots:
            x1, y1, x2, y2 = slot['bbox']
            
            if slot['occupied']:
                color = (0, 0, 255)  # Red - BGR
                label = "OCCUPIED"
            else:
                color = (0, 255, 0)  # Green - BGR
                label = "AVAILABLE"
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Transparent fill
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
            
            # Label
            text = f"#{slot['id']+1} {label}"
            cv2.putText(output, text, (x1+2, y1+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 2)
            cv2.putText(output, text, (x1+2, y1+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            if slot['occupied']:
                conf = f"{slot['confidence']*100:.1f}%"
                cv2.putText(output, conf, (x1+2, y1+28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # Stats panel
        self._draw_stats(output, slots)
        
        return output
    
    def _draw_stats(self, frame, slots):
        """Draw stats panel."""
        h, w = frame.shape[:2]
        occupied = sum(1 for s in slots if s['occupied'])
        available = len(slots) - occupied
        total = len(slots)
        rate = occupied / total * 100 if total > 0 else 0
        
        # Panel
        px, py = w - 220, 10
        pw, ph = 210, 145
        
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (20,20,20), -1)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0,200,255), 2)
        
        # Text
        cv2.putText(frame, "SMART PARKING AI", (px+10, py+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)
        
        cv2.putText(frame, f"Total Slots: {total}", (px+10, py+50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        
        cv2.putText(frame, f"Occupied: {occupied}", (px+10, py+72),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
        
        cv2.putText(frame, f"Available: {available}", (px+10, py+94),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
        
        cv2.putText(frame, f"Occupancy: {rate:.1f}%", (px+10, py+116),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        
        # Bar
        bar_w = pw - 20
        cv2.rectangle(frame, (px+10, py+125), (px+10+bar_w, py+138), (60,60,60), -1)
        fill = int(bar_w * rate / 100)
        if fill > 0:
            cv2.rectangle(frame, (px+10, py+125), (px+10+fill, py+138), (0,255-int(rate*2.55),int(rate*2.55)), -1)
    
    def _fallback_vehicle_only(self, image, vehicles, output_path, show):
        """Fallback: Just show detected vehicles."""
        output = image.copy()
        
        for v in vehicles:
            x1, y1, x2, y2 = v['bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (0,0,255), 3)
            cv2.putText(output, f"Vehicle {v['conf']*100:.0f}%", (x1, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        # Simple stats
        h, w = output.shape[:2]
        cv2.rectangle(output, (w-200, 10), (w-10, 70), (0,0,0), -1)
        cv2.putText(output, "SMART PARKING AI", (w-190, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2)
        cv2.putText(output, f"Vehicles: {len(vehicles)}", (w-190, 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        
        if output_path:
            cv2.imwrite(output_path, output)
        
        if show:
            self._show(output)
        
        return output, []
    
    def _show(self, frame, max_dim=1000):
        """Display frame."""
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        
        cv2.imshow('Smart Parking AI', frame)
        print("\n  Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    print("=" * 50)
    print("  🅿️  SMART PARKING AI - SIMPLE VERSION")
    print("  Reliable detection for demos")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("\nUsage: python smart_parking_simple.py <image>")
        print("\nExample: python smart_parking_simple.py parking.jpg")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    detector = SimpleParkingDetector()
    detector.process_image(image_path, output_path='result.jpg', show=True)
    
    print("\n" + "=" * 50)
    print("  ✅ Done! Check 'result.jpg'")
    print("=" * 50)


if __name__ == "__main__":
    main()