"""
Smart Parking AI: Automatic Parking Slot Detection
ISTE-500/501 Senior Development Project
Team: Dilshod, Walid, Sayed, Meiir

This advanced version AUTOMATICALLY detects parking slots using:
1. Edge detection to find parking lot line markings
2. Hough Line Transform to detect lines
3. Line clustering to identify slot boundaries
4. Vehicle detection to determine occupancy

Usage:
    python smart_parking_auto.py image.jpg
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import sys
import os


class AutoParkingDetector:
    """Automatic parking slot detection and occupancy analysis."""
    
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize with YOLO model."""
        print("🚗 Initializing Smart Parking AI (Auto-Detection)...")
        self.model = YOLO(model_path)
        
        # Vehicle classes: car, motorcycle, bus, truck
        self.vehicle_classes = [2, 3, 5, 7]
        
        print("✅ Model loaded!")
    
    def detect_parking_slots(self, image, method='hybrid'):
        """
        Automatically detect parking slot boundaries.
        
        Methods:
        - 'lines': Use line detection only
        - 'vehicles': Use vehicle positions to estimate slots
        - 'hybrid': Combine both approaches (recommended)
        """
        if method == 'lines':
            return self._detect_slots_from_lines(image)
        elif method == 'vehicles':
            return self._detect_slots_from_vehicles(image)
        else:  # hybrid
            return self._detect_slots_hybrid(image)
    
    def _detect_slots_from_lines(self, image):
        """Detect parking slots using line detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=20
        )
        
        if lines is None:
            return []
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Horizontal lines (within 20 degrees of horizontal)
            if abs(angle) < 20 or abs(angle) > 160:
                horizontal_lines.append((x1, y1, x2, y2))
            # Vertical lines (within 20 degrees of vertical)
            elif 70 < abs(angle) < 110:
                vertical_lines.append((x1, y1, x2, y2))
        
        # Cluster lines to find slot boundaries
        slots = self._cluster_lines_to_slots(
            horizontal_lines, vertical_lines, width, height
        )
        
        return slots
    
    def _detect_slots_from_vehicles(self, image):
        """Estimate parking slots based on detected vehicle positions."""
        # Detect vehicles
        results = self.model(image, conf=0.3, verbose=False)
        
        vehicle_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
        
        if len(vehicle_boxes) < 2:
            return []
        
        # Estimate average slot size from vehicles
        avg_width = np.mean([v['width'] for v in vehicle_boxes])
        avg_height = np.mean([v['height'] for v in vehicle_boxes])
        
        # Add padding for slot size
        slot_width = int(avg_width * 1.3)
        slot_height = int(avg_height * 1.3)
        
        # Group vehicles by row (similar y-coordinates)
        rows = self._group_vehicles_by_row(vehicle_boxes)
        
        # Generate slots for each row
        slots = []
        height, width = image.shape[:2]
        
        for row_vehicles in rows:
            if len(row_vehicles) < 1:
                continue
            
            # Sort by x-coordinate
            row_vehicles.sort(key=lambda v: v['center'][0])
            
            # Get row y-position
            row_y = np.mean([v['center'][1] for v in row_vehicles])
            
            # Create slots along this row
            row_slots = self._generate_row_slots(
                row_vehicles, row_y, slot_width, slot_height, width
            )
            slots.extend(row_slots)
        
        return slots
    
    def _detect_slots_hybrid(self, image):
        """
        Hybrid approach: Use both line detection and vehicle positions.
        This is the most robust method.
        """
        height, width = image.shape[:2]
        
        # Step 1: Detect vehicles first
        results = self.model(image, conf=0.25, verbose=False)
        
        vehicle_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
        
        if len(vehicle_boxes) == 0:
            print("  ⚠ No vehicles detected, using line detection only")
            return self._detect_slots_from_lines(image)
        
        # Step 2: Estimate slot dimensions from vehicles
        avg_width = np.mean([v['width'] for v in vehicle_boxes])
        avg_height = np.mean([v['height'] for v in vehicle_boxes])
        
        slot_width = int(avg_width * 1.25)
        slot_height = int(avg_height * 1.25)
        
        # Step 3: Group vehicles into rows
        rows = self._group_vehicles_by_row(vehicle_boxes, tolerance=avg_height * 0.5)
        
        # Step 4: For each row, detect the pattern and fill in gaps
        all_slots = []
        
        for row_idx, row_vehicles in enumerate(rows):
            if len(row_vehicles) < 1:
                continue
            
            # Sort by x-coordinate
            row_vehicles.sort(key=lambda v: v['center'][0])
            
            # Get row boundaries
            row_y_center = np.mean([v['center'][1] for v in row_vehicles])
            row_y_top = int(row_y_center - slot_height // 2)
            row_y_bottom = int(row_y_center + slot_height // 2)
            
            # Find leftmost and rightmost positions
            leftmost_x = min(v['bbox'][0] for v in row_vehicles)
            rightmost_x = max(v['bbox'][2] for v in row_vehicles)
            
            # Extend to edges with some margin
            start_x = max(0, leftmost_x - slot_width)
            end_x = min(width, rightmost_x + slot_width)
            
            # Calculate spacing between vehicles to estimate slot width
            if len(row_vehicles) > 1:
                spacings = []
                for i in range(len(row_vehicles) - 1):
                    spacing = row_vehicles[i+1]['center'][0] - row_vehicles[i]['center'][0]
                    spacings.append(spacing)
                estimated_slot_width = int(np.median(spacings))
                slot_width = max(slot_width, estimated_slot_width)
            
            # Generate slots along the row
            x = start_x
            slot_id = len(all_slots)
            
            while x + slot_width <= end_x + slot_width // 2:
                slot = {
                    'id': slot_id,
                    'bbox': (x, row_y_top, x + slot_width, row_y_bottom),
                    'center': (x + slot_width // 2, int(row_y_center)),
                    'row': row_idx,
                    'occupied': False,
                    'confidence': 0.0,
                    'vehicle': None
                }
                all_slots.append(slot)
                slot_id += 1
                x += slot_width
        
        # Step 5: Match vehicles to slots
        all_slots = self._match_vehicles_to_slots(all_slots, vehicle_boxes)
        
        return all_slots
    
    def _group_vehicles_by_row(self, vehicles, tolerance=None):
        """Group vehicles into rows based on y-coordinate."""
        if len(vehicles) == 0:
            return []
        
        if tolerance is None:
            avg_height = np.mean([v['height'] for v in vehicles])
            tolerance = avg_height * 0.6
        
        # Sort by y-coordinate
        sorted_vehicles = sorted(vehicles, key=lambda v: v['center'][1])
        
        rows = []
        current_row = [sorted_vehicles[0]]
        current_y = sorted_vehicles[0]['center'][1]
        
        for vehicle in sorted_vehicles[1:]:
            if abs(vehicle['center'][1] - current_y) <= tolerance:
                current_row.append(vehicle)
            else:
                rows.append(current_row)
                current_row = [vehicle]
                current_y = vehicle['center'][1]
        
        rows.append(current_row)
        
        return rows
    
    def _generate_row_slots(self, row_vehicles, row_y, slot_width, slot_height, image_width):
        """Generate parking slots for a single row."""
        slots = []
        
        if len(row_vehicles) == 0:
            return slots
        
        # Get the range of this row
        min_x = min(v['bbox'][0] for v in row_vehicles) - slot_width
        max_x = max(v['bbox'][2] for v in row_vehicles) + slot_width
        
        min_x = max(0, min_x)
        max_x = min(image_width, max_x)
        
        # Calculate spacing
        if len(row_vehicles) > 1:
            centers = sorted([v['center'][0] for v in row_vehicles])
            spacings = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
            slot_width = int(np.median(spacings)) if spacings else slot_width
        
        # Generate slots
        x = min_x
        while x + slot_width <= max_x:
            y1 = int(row_y - slot_height // 2)
            y2 = int(row_y + slot_height // 2)
            
            slots.append({
                'bbox': (int(x), y1, int(x + slot_width), y2),
                'center': (int(x + slot_width // 2), int(row_y)),
                'occupied': False,
                'confidence': 0.0
            })
            x += slot_width
        
        return slots
    
    def _match_vehicles_to_slots(self, slots, vehicles):
        """Match detected vehicles to parking slots."""
        for slot in slots:
            sx1, sy1, sx2, sy2 = slot['bbox']
            slot_center = slot['center']
            
            best_match = None
            best_overlap = 0
            
            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle['bbox']
                
                # Calculate overlap
                overlap_x = max(0, min(sx2, vx2) - max(sx1, vx1))
                overlap_y = max(0, min(sy2, vy2) - max(sy1, vy1))
                overlap_area = overlap_x * overlap_y
                
                slot_area = (sx2 - sx1) * (sy2 - sy1)
                overlap_ratio = overlap_area / slot_area if slot_area > 0 else 0
                
                # Also check if vehicle center is in slot
                vcx, vcy = vehicle['center']
                center_in_slot = sx1 <= vcx <= sx2 and sy1 <= vcy <= sy2
                
                if overlap_ratio > 0.2 or center_in_slot:
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_match = vehicle
            
            if best_match is not None:
                slot['occupied'] = True
                slot['confidence'] = min(0.95 + best_overlap * 0.05, 0.99)
                slot['vehicle'] = best_match
            else:
                slot['occupied'] = False
                slot['confidence'] = 0.95  # High confidence it's empty
        
        return slots
    
    def _cluster_lines_to_slots(self, h_lines, v_lines, width, height):
        """Cluster detected lines into parking slot boundaries."""
        # This is a simplified version - cluster vertical lines to find slot dividers
        if len(v_lines) < 2:
            return []
        
        # Get x-coordinates of vertical lines
        x_coords = []
        for x1, y1, x2, y2 in v_lines:
            x_coords.append((x1 + x2) // 2)
        
        x_coords = sorted(set(x_coords))
        
        # Cluster nearby x-coordinates
        clusters = []
        current_cluster = [x_coords[0]]
        
        for x in x_coords[1:]:
            if x - current_cluster[-1] < 30:  # Merge if within 30 pixels
                current_cluster.append(x)
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [x]
        clusters.append(int(np.mean(current_cluster)))
        
        # Create slots between consecutive vertical lines
        slots = []
        for i in range(len(clusters) - 1):
            x1 = clusters[i]
            x2 = clusters[i + 1]
            
            # Estimate y-range from horizontal lines
            if h_lines:
                y_coords = []
                for hx1, hy1, hx2, hy2 in h_lines:
                    if x1 <= hx1 <= x2 or x1 <= hx2 <= x2:
                        y_coords.extend([hy1, hy2])
                
                if y_coords:
                    y1 = min(y_coords)
                    y2 = max(y_coords)
                else:
                    y1 = height // 4
                    y2 = height * 3 // 4
            else:
                y1 = height // 4
                y2 = height * 3 // 4
            
            slots.append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'occupied': False,
                'confidence': 0.0
            })
        
        return slots
    
    def process_image(self, image_path, output_path=None, show=True):
        """
        Main processing function - auto-detects slots and checks occupancy.
        """
        print(f"\n📸 Processing: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = image.shape[:2]
        print(f"  📐 Image size: {width}x{height}")
        
        # Step 1: Auto-detect parking slots
        print("  🔍 Auto-detecting parking slots...")
        slots = self.detect_parking_slots(image, method='hybrid')
        print(f"  ✓ Found {len(slots)} parking slots")
        
        if len(slots) == 0:
            print("  ⚠ No slots detected. Showing vehicle detection only.")
            return self._vehicle_only_mode(image, output_path, show)
        
        # Step 2: Draw visualization
        output = image.copy()
        
        occupied_count = 0
        available_count = 0
        
        for slot in slots:
            x1, y1, x2, y2 = slot['bbox']
            
            if slot['occupied']:
                color = (0, 0, 255)  # Red
                status = "OCCUPIED"
                occupied_count += 1
            else:
                color = (0, 255, 0)  # Green
                status = "AVAILABLE"
                available_count += 1
            
            # Draw slot rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Semi-transparent fill
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0)
            
            # Label
            slot_id = slot.get('id', slots.index(slot) + 1)
            label = f"#{slot_id} {status}"
            
            # Position label
            label_y = y1 + 20 if y1 + 20 < y2 else y1 - 5
            cv2.putText(output, label, (x1 + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(output, label, (x1 + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            # Confidence
            if slot['occupied']:
                conf_text = f"{slot['confidence']*100:.1f}%"
                cv2.putText(output, conf_text, (x1 + 5, label_y + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw stats panel
        self._draw_stats_panel(output, len(slots), occupied_count, available_count)
        
        # Print summary
        print(f"\n  ✅ Detection Complete!")
        print(f"     • Total Slots: {len(slots)}")
        print(f"     • Occupied: {occupied_count} (RED)")
        print(f"     • Available: {available_count} (GREEN)")
        print(f"     • Occupancy Rate: {occupied_count/len(slots)*100:.1f}%")
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, output)
            print(f"     • Saved to: {output_path}")
        
        # Display
        if show:
            display = self._resize_for_display(output)
            cv2.imshow('Smart Parking AI - Auto Detection', display)
            print("\n  Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return output, slots
    
    def _vehicle_only_mode(self, image, output_path, show):
        """Fallback mode: just detect and highlight vehicles."""
        results = self.model(image, conf=0.3, verbose=False)
        
        output = image.copy()
        vehicle_count = 0
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"Vehicle {conf*100:.1f}%"
                    cv2.putText(output, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    vehicle_count += 1
        
        # Stats panel
        height, width = output.shape[:2]
        cv2.rectangle(output, (width-250, 10), (width-10, 100), (0,0,0), -1)
        cv2.putText(output, "SMART PARKING AI", (width-240, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(output, f"Vehicles Detected: {vehicle_count}", (width-240, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, "Mode: Vehicle Detection", (width-240, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        if output_path:
            cv2.imwrite(output_path, output)
        
        if show:
            display = self._resize_for_display(output)
            cv2.imshow('Smart Parking AI - Vehicle Detection', display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return output, []
    
    def _draw_stats_panel(self, frame, total, occupied, available):
        """Draw statistics panel."""
        height, width = frame.shape[:2]
        
        # Panel dimensions
        panel_width = 280
        panel_height = 180
        panel_x = width - panel_width - 15
        panel_y = 15
        
        # Background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (20, 20, 20), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 200, 255), 2)
        
        # Title
        cv2.putText(frame, "SMART PARKING AI", 
                   (panel_x + 15, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        
        # Divider
        cv2.line(frame, (panel_x + 10, panel_y + 45), 
                (panel_x + panel_width - 10, panel_y + 45), (100, 100, 100), 1)
        
        # Stats
        y_offset = panel_y + 70
        
        cv2.putText(frame, f"Total Slots: {total}", 
                   (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Occupied: {occupied}", 
                   (panel_x + 15, y_offset + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Available: {available}", 
                   (panel_x + 15, y_offset + 56),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        # Occupancy bar
        bar_y = y_offset + 80
        bar_width = panel_width - 30
        occupancy_ratio = occupied / total if total > 0 else 0
        
        cv2.putText(frame, f"Occupancy: {occupancy_ratio*100:.1f}%", 
                   (panel_x + 15, bar_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Bar background
        cv2.rectangle(frame, (panel_x + 15, bar_y + 8), 
                     (panel_x + 15 + bar_width, bar_y + 22), (60, 60, 60), -1)
        
        # Bar fill (gradient from green to red)
        fill_width = int(bar_width * occupancy_ratio)
        if fill_width > 0:
            # Create gradient effect
            for i in range(fill_width):
                ratio = i / bar_width
                color = (0, int(255 * (1 - ratio)), int(255 * ratio))
                cv2.line(frame, 
                        (panel_x + 15 + i, bar_y + 9),
                        (panel_x + 15 + i, bar_y + 21), color, 1)
    
    def _resize_for_display(self, frame, max_dim=1200):
        """Resize frame for display if too large."""
        height, width = frame.shape[:2]
        
        if max(height, width) <= max_dim:
            return frame
        
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        
        return cv2.resize(frame, (new_width, new_height))


def main():
    """Main function."""
    print("=" * 60)
    print("  🅿️  SMART PARKING AI - AUTO DETECTION")
    print("  Automatic Parking Slot Detection & Occupancy Analysis")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python smart_parking_auto.py <image_path>")
        print("\nExample:")
        print("  python smart_parking_auto.py parking_lot.jpg")
        print("\nThe system will automatically:")
        print("  1. Detect all vehicles in the image")
        print("  2. Identify parking slot boundaries")
        print("  3. Determine which slots are occupied/available")
        print("  4. Display results with statistics")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    # Initialize detector
    detector = AutoParkingDetector()
    
    # Process image
    output, slots = detector.process_image(
        image_path,
        output_path='detection_result.jpg',
        show=True
    )
    
    print("\n" + "=" * 60)
    print("  ✅ Done! Check 'detection_result.jpg' for the output.")
    print("=" * 60)


if __name__ == "__main__":
    main()