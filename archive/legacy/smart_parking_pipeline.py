"""
Smart Parking AI: Auto-calibration + live occupancy pipeline.

This module builds on the existing line-based detector to:
1) Auto-discover slots from a set of frames (no manual JSON)
2) Persist the discovered slots to slots_auto.json
3) Process live frames to mark slot occupancy with smoothing
"""

from __future__ import annotations

import json
from statistics import median
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

from smart_parking_lines import LineParkingDetector


class ParkingSlotPipeline:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        overlap_threshold: float = 0.15,
        confirm_frames: int = 3,
        auto_scale: bool = True,
    ) -> None:
        self.detector = LineParkingDetector(model_path=model_path)
        self.overlap_threshold = overlap_threshold
        self.confirm_frames = max(1, int(confirm_frames))
        self.auto_scale = auto_scale

        self._slot_state: Dict[int, Dict[str, int]] = {}
        self._slots: Optional[List[Dict]] = None
        self._slot_shape: Optional[Tuple[int, int]] = None

    # ---------------------------
    # Calibration
    # ---------------------------
    def calibrate_from_frames(
        self,
        frames: List[np.ndarray],
        save_path: str = "slots_auto.json",
        min_slots: int = 10,
        max_slots: int = 60,
        mode: str = "auto",
    ) -> List[Dict]:
        if not frames:
            raise ValueError("No frames provided for calibration.")

        mode = mode.lower().strip()
        if mode not in {"auto", "vehicle", "lines", "projection", "vehicle_gaps"}:
            raise ValueError("mode must be one of: auto, vehicle, lines, projection, vehicle_gaps")

        if mode == "vehicle":
            best_slots = self._slots_from_vehicle_history(frames)
            if not best_slots:
                # Fallback to line dividers if vehicle detection is weak.
                best_slots = self._slots_from_line_dividers(frames)
            if not best_slots:
                raise RuntimeError("Vehicle-grid calibration failed (no slots).")
            best_slots = self._reindex_slots(best_slots)
            self._slots = best_slots
            self._slot_shape = frames[0].shape[:2]
            self._slot_state = {}
            self.save_slots(best_slots, self._slot_shape, save_path)
            return best_slots

        if mode == "vehicle_gaps":
            best_slots = self._slots_from_vehicle_gaps(frames)
            if not best_slots:
                raise RuntimeError("Vehicle-gaps calibration failed (no slots).")
            best_slots = self._reindex_slots(best_slots)
            self._slots = best_slots
            self._slot_shape = frames[0].shape[:2]
            self._slot_state = {}
            self.save_slots(best_slots, self._slot_shape, save_path)
            return best_slots

        if mode == "lines":
            best_slots = self._slots_from_line_dividers(frames)
            if not best_slots:
                raise RuntimeError("Line-divider calibration failed (no slots).")
            best_slots = self._reindex_slots(best_slots)
            self._slots = best_slots
            self._slot_shape = frames[0].shape[:2]
            self._slot_state = {}
            self.save_slots(best_slots, self._slot_shape, save_path)
            return best_slots

        if mode == "projection":
            best_slots = self._slots_from_projection(frames)
            if not best_slots:
                raise RuntimeError("Projection calibration failed (no slots).")
            best_slots = self._reindex_slots(best_slots)
            self._slots = best_slots
            self._slot_shape = frames[0].shape[:2]
            self._slot_state = {}
            self.save_slots(best_slots, self._slot_shape, save_path)
            return best_slots

        candidates = []
        all_nonempty = []
        for frame in frames:
            slots = self._detect_slots_in_frame(frame, apply_filter=False)
            if slots:
                all_nonempty.append((frame, slots))
                if min_slots <= len(slots) <= max_slots:
                    candidates.append((frame, slots))

        if not candidates:
            # Fall back to any non-empty candidate before giving up.
            candidates = all_nonempty

        if not candidates:
            raise RuntimeError("No valid slot candidates found during calibration.")

        # Prefer the candidate with the most slots, but penalize wild geometry variance.
        best_slots = None
        best_score = None
        for _, slots in candidates:
            count = len(slots)
            score = (-count) + (self._slot_geometry_variation(slots) * 2.0)
            if best_score is None or score < best_score:
                best_score = score
                best_slots = slots

        assert best_slots is not None

        # If we still have too few slots, build a grid from vehicle history.
        if len(best_slots) < min_slots:
            grid_slots = self._slots_from_vehicle_history(frames)
            if grid_slots:
                best_slots = grid_slots
            else:
                line_slots = self._slots_from_line_dividers(frames)
                if line_slots:
                    best_slots = line_slots

        best_slots = self._reindex_slots(best_slots)

        self._slots = best_slots
        self._slot_shape = frames[0].shape[:2]
        self._slot_state = {}

        self.save_slots(best_slots, self._slot_shape, save_path)
        return best_slots

    # ---------------------------
    # Live processing
    # ---------------------------
    def load_slots(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            data = json.load(f)

        slots = data.get("slots", data)
        shape = data.get("image_shape")
        if shape:
            self._slot_shape = (int(shape[0]), int(shape[1]))
        else:
            self._slot_shape = None

        self._slots = self._reindex_slots(slots)
        self._slot_state = {}
        return self._slots

    def process_frame(self, frame: np.ndarray) -> Dict:
        if self._slots is None:
            raise RuntimeError("Slots not loaded. Run calibrate or load_slots first.")

        slots = self._slots
        if self.auto_scale and self._slot_shape is not None:
            slots = self._scale_slots_if_needed(slots, self._slot_shape, frame.shape[:2])

        vehicles = self.detector.detect_vehicles(frame)
        slots = self._assign_occupancy(slots, vehicles)
        slots = self._apply_smoothing(slots)

        occupied = sum(1 for s in slots if s["occupied"])
        available = len(slots) - occupied

        return {
            "total": len(slots),
            "occupied": occupied,
            "available": available,
            "slots": slots,
        }

    def annotate_frame(self, frame: np.ndarray, slots: List[Dict]) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]

        for slot in slots:
            x1, y1, x2, y2 = slot["bbox"]
            x1 = max(0, min(int(x1), width - 1))
            x2 = max(0, min(int(x2), width - 1))
            y1 = max(0, min(int(y1), height - 1))
            y2 = max(0, min(int(y2), height - 1))

            if slot["occupied"]:
                color = (0, 0, 255)
                status = "OCCUPIED"
            else:
                color = (0, 255, 0)
                status = "AVAILABLE"

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0)

            label = f"#{slot['id']+1} {status}"
            cv2.putText(
                output,
                label,
                (x1 + 3, max(12, y1 + 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                output,
                label,
                (x1 + 3, max(12, y1 + 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return output

    @staticmethod
    def save_slots(slots: List[Dict], image_shape: Tuple[int, int], path: str) -> None:
        payload = {
            "image_shape": [int(image_shape[0]), int(image_shape[1])],
            "slots": slots,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _detect_slots_in_frame(self, image: np.ndarray, apply_filter: bool = True) -> List[Dict]:
        height, width = image.shape[:2]

        vehicles = self.detector.detect_vehicles(image)
        white_mask = self.detector.detect_white_lines(image)
        lines = self.detector.detect_line_segments(white_mask)
        vertical_lines, horizontal_lines = self.detector.classify_lines(lines, image.shape)

        all_slot_methods = []

        # Method 1: Connected components
        image_area = width * height
        if len(vehicles) > 0:
            avg_vehicle_area = np.mean(
                [(v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]) for v in vehicles]
            )
            min_slot_area = max(int(image_area / 1200), int(avg_vehicle_area * 0.2))
            max_slot_area = min(int(image_area / 4), int(avg_vehicle_area * 6.0))
        else:
            min_slot_area = int(image_area / 1200)
            max_slot_area = int(image_area / 4)

        cc_slots = self.detector.detect_slots_connected_components(
            white_mask, image.shape, min_slot_area, max_slot_area, debug=False
        )
        if len(cc_slots) > 0:
            all_slot_methods.append(("connected_components", cc_slots))

        # Method 2: Contours
        contour_slots = self.detector.detect_slots_from_contours(white_mask, image.shape, vehicles)
        if len(contour_slots) > 0:
            all_slot_methods.append(("contours", contour_slots))

        # Method 3: Vertical line-based
        if len(vertical_lines) > 10:
            clustered_lines = self.detector.cluster_vertical_lines(vertical_lines)
            rows = self.detector._detect_rows_from_vehicles(vehicles, clustered_lines, width, height)
            if rows and rows[0].get("lines"):
                line_slots = self.detector.create_slots_from_rows(rows, width, height)
            else:
                line_slots = self.detector._create_slots_for_rows(rows, clustered_lines, width, height)
            if len(line_slots) > 0:
                all_slot_methods.append(("vertical_lines", line_slots))

        # Method 4: Horizontal line-based
        if len(horizontal_lines) > len(vertical_lines) * 2:
            horiz_slots = self.detector.detect_slots_from_horizontal_lines(
                horizontal_lines, vertical_lines, vehicles, width, height
            )
            if len(horiz_slots) > 0:
                all_slot_methods.append(("horizontal_lines", horiz_slots))

        slots = self._select_best_method(all_slot_methods, vehicles)

        if len(slots) == 0 and len(vehicles) >= 2:
            slots = self._fallback_vehicle_grid(vehicles, width, height)

        if len(slots) > 0:
            if apply_filter:
                slots = self.detector._filter_slots(slots, vehicles, (height, width))
            slots = self.detector._assign_row_numbers(slots, height)
            if apply_filter:
                slots = self.detector._merge_slots_by_row(slots, (height, width))

        return self._reindex_slots(slots)

    @staticmethod
    def _select_best_method(methods, vehicles):
        if not methods:
            return []

        best_score = -1.0
        best = []

        for _, slots in methods:
            n_slots = len(slots)
            score = float(n_slots)

            if vehicles:
                ratio = n_slots / max(len(vehicles), 1)
                if 0.8 <= ratio <= 2.5:
                    score *= 2.0
                elif 2.5 < ratio <= 3.5:
                    score *= 1.3
                elif 3.5 < ratio <= 5.0:
                    score *= 0.7
                else:
                    score *= 0.4

            if n_slots < 5:
                score *= 0.4
            if n_slots > 120:
                score *= 0.2

            if score > best_score:
                best_score = score
                best = slots

        return best

    @staticmethod
    def _fallback_vehicle_grid(vehicles, width, height) -> List[Dict]:
        # Simple grid derived from vehicle locations
        centers = sorted([v["center"] for v in vehicles], key=lambda c: c[1])
        if len(centers) < 2:
            return []

        widths = [v["bbox"][2] - v["bbox"][0] for v in vehicles]
        heights = [v["bbox"][3] - v["bbox"][1] for v in vehicles]
        avg_w = int(np.median(widths))
        avg_h = int(np.median(heights))
        slot_w = int(avg_w * 1.25)
        slot_h = int(avg_h * 1.25)

        # Group into rows by Y
        rows = []
        tolerance = avg_h * 0.7
        current = [vehicles[0]]
        current_y = vehicles[0]["center"][1]

        for v in sorted(vehicles, key=lambda vv: vv["center"][1])[1:]:
            if abs(v["center"][1] - current_y) <= tolerance:
                current.append(v)
                current_y = np.mean([vv["center"][1] for vv in current])
            else:
                rows.append(current)
                current = [v]
                current_y = v["center"][1]
        rows.append(current)

        slots = []
        slot_id = 0
        for row_idx, row_vehicles in enumerate(rows):
            xs = []
            for v in row_vehicles:
                xs.extend([v["bbox"][0], v["bbox"][2]])
            if not xs:
                continue
            min_x = min(xs)
            max_x = max(xs)
            row_y = int(np.median([v["center"][1] for v in row_vehicles]))
            y1 = max(0, row_y - slot_h // 2)
            y2 = min(height, row_y + slot_h // 2)

            row_start = max(0, min_x - slot_w * 2)
            row_end = min(width, max_x + slot_w * 2)

            x = row_start
            while x + slot_w <= row_end:
                slots.append(
                    {
                        "id": slot_id,
                        "bbox": (int(x), int(y1), int(x + slot_w), int(y2)),
                        "center": (int(x + slot_w // 2), int((y1 + y2) // 2)),
                        "row": row_idx,
                        "occupied": False,
                        "confidence": 0.0,
                        "vehicle": None,
                    }
                )
                slot_id += 1
                x += slot_w

        return slots

    def _slots_from_vehicle_history(self, frames: List[np.ndarray]) -> List[Dict]:
        vehicles_all = []
        widths = []
        heights = []
        for frame in frames:
            vehicles = self.detector.detect_vehicles(frame)
            for v in vehicles:
                vehicles_all.append(v)
                widths.append(v["bbox"][2] - v["bbox"][0])
                heights.append(v["bbox"][3] - v["bbox"][1])

        if len(vehicles_all) < 4:
            return []

        img_h, img_w = frames[0].shape[:2]
        med_w = int(np.median(widths))
        med_h = int(np.median(heights))
        slot_w_base = int(med_w * 1.25)
        slot_h = int(med_h * 1.25)

        # Cluster vehicles into rows using DBSCAN for stability
        from sklearn.cluster import DBSCAN

        y_centers = np.array([[v["center"][1]] for v in vehicles_all])
        row_eps = max(10, int(med_h * 0.6))
        clustering = DBSCAN(eps=row_eps, min_samples=2).fit(y_centers)

        rows_dict = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue
            rows_dict.setdefault(label, []).append(vehicles_all[idx])

        rows = list(rows_dict.values())
        rows.sort(key=lambda r: np.mean([v["center"][1] for v in r]))

        slots = []
        slot_id = 0
        for row_idx, row_vehicles in enumerate(rows):
            if len(row_vehicles) < 2:
                continue

            centers_x = sorted([v["center"][0] for v in row_vehicles])
            spacings = [centers_x[i + 1] - centers_x[i] for i in range(len(centers_x) - 1)]
            med_spacing = int(np.median(spacings)) if spacings else slot_w_base
            slot_w_row = max(int(slot_w_base * 0.6), min(int(slot_w_base * 1.6), med_spacing))

            row_y = int(np.median([v["center"][1] for v in row_vehicles]))
            y1 = max(0, row_y - slot_h // 2)
            y2 = min(img_h, row_y + slot_h // 2)

            min_x = min([v["bbox"][0] for v in row_vehicles])
            max_x = max([v["bbox"][2] for v in row_vehicles])

            row_start = max(0, min_x - slot_w_row * 4)
            row_end = min(img_w, max_x + slot_w_row * 4)

            x = row_start
            while x + slot_w_row <= row_end:
                slots.append(
                    {
                        "id": slot_id,
                        "bbox": (int(x), int(y1), int(x + slot_w_row), int(y2)),
                        "center": (int(x + slot_w_row // 2), int((y1 + y2) // 2)),
                        "row": row_idx,
                        "occupied": False,
                        "confidence": 0.0,
                        "vehicle": None,
                    }
                )
                slot_id += 1
                x += slot_w_row

        return slots

    def _slots_from_line_dividers(self, frames: List[np.ndarray]) -> List[Dict]:
        if not frames:
            return []

        img_h, img_w = frames[0].shape[:2]

        # Collect vehicle stats to estimate slot height and row bands
        vehicles_all = []
        heights = []
        for frame in frames:
            vehicles = self.detector.detect_vehicles(frame)
            vehicles_all.extend(vehicles)
            for v in vehicles:
                heights.append(v["bbox"][3] - v["bbox"][1])

        use_vehicle_rows = len(vehicles_all) >= 4

        med_h = int(np.median(heights)) if heights else max(20, img_h // 12)
        slot_h = int(med_h * 1.25)
        row_tol = max(10, int(med_h * 0.7))

        # Detect vertical line dividers across frames
        all_x = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=60,
                minLineLength=max(20, img_h // 15),
                maxLineGap=30,
            )
            if lines is None:
                continue
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    angle = 90
                else:
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if 75 <= angle <= 105:  # near-vertical dividers
                    all_x.append((x1 + x2) / 2)

        if len(all_x) < 4:
            return []

        # Cluster divider x-positions
        from sklearn.cluster import DBSCAN

        x_array = np.array([[x] for x in all_x])
        clustering = DBSCAN(eps=max(12, img_w // 120), min_samples=3).fit(x_array)
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(all_x[idx])
        dividers = sorted([int(np.mean(xs)) for xs in clusters.values()])

        if len(dividers) < 3:
            return []

        rows = []
        # Prefer horizontal line bands for robust row detection
        all_y = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=60,
                minLineLength=max(20, img_w // 12),
                maxLineGap=25,
            )
            if lines is None:
                continue
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    angle = 90
                else:
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle <= 12 or angle >= 168:  # near-horizontal
                    all_y.append((y1 + y2) / 2)

        band_centers = []
        if len(all_y) >= 6:
            from sklearn.cluster import DBSCAN

            y_array = np.array([[y] for y in all_y])
            y_eps = max(8, img_h // 50)
            y_cluster = DBSCAN(eps=y_eps, min_samples=3).fit(y_array)
            y_clusters = {}
            for idx, label in enumerate(y_cluster.labels_):
                if label == -1:
                    continue
                y_clusters.setdefault(label, []).append(all_y[idx])
            band_centers = sorted([int(np.mean(ys)) for ys in y_clusters.values()])

        if not band_centers and use_vehicle_rows:
            # Build row centers from vehicle Y clusters
            vehicles_sorted = sorted(vehicles_all, key=lambda v: v["center"][1])
            current = [vehicles_sorted[0]]
            current_y = vehicles_sorted[0]["center"][1]
            for v in vehicles_sorted[1:]:
                if abs(v["center"][1] - current_y) <= row_tol:
                    current.append(v)
                    current_y = int(np.mean([vv["center"][1] for vv in current]))
                else:
                    rows.append(current)
                    current = [v]
                    current_y = v["center"][1]
            rows.append(current)
        else:
            # Convert horizontal bands into row centers by taking midpoints
            band_centers = sorted(band_centers)
            if len(band_centers) >= 2:
                for i in range(len(band_centers) - 1):
                    y_mid = (band_centers[i] + band_centers[i + 1]) // 2
                    rows.append([{"center": (0, y_mid)}] * 2)
            elif band_centers:
                rows.append([{"center": (0, band_centers[0])}] * 2)
            elif use_vehicle_rows and not rows:
                rows = [[{"center": (0, img_h // 2)}] * 2]

        slots = []
        slot_id = 0
        for row_idx, row_vehicles in enumerate(rows):
            if len(row_vehicles) < 2:
                continue
            row_y = int(np.median([v["center"][1] for v in row_vehicles]))
            y1 = max(0, row_y - slot_h // 2)
            y2 = min(img_h, row_y + slot_h // 2)

            # Create slots between consecutive dividers
            for i in range(len(dividers) - 1):
                x1 = dividers[i]
                x2 = dividers[i + 1]
                if x2 - x1 < 15:
                    continue
                slots.append(
                    {
                        "id": slot_id,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "center": (int((x1 + x2) // 2), int((y1 + y2) // 2)),
                        "row": row_idx,
                        "occupied": False,
                        "confidence": 0.0,
                        "vehicle": None,
                    }
                )
                slot_id += 1

        return slots

    def _slots_from_projection(self, frames: List[np.ndarray]) -> List[Dict]:
        if not frames:
            return []

        h, w = frames[0].shape[:2]

        # Estimate slot size from vehicles if possible
        veh_w = []
        veh_h = []
        for frame in frames:
            vehicles = self.detector.detect_vehicles(frame)
            for v in vehicles:
                veh_w.append(v["bbox"][2] - v["bbox"][0])
                veh_h.append(v["bbox"][3] - v["bbox"][1])
        est_slot_w = int(np.median(veh_w)) if veh_w else max(25, w // 20)
        est_slot_h = int(np.median(veh_h)) if veh_h else max(20, h // 15)

        # Aggregate white-line masks to stabilize
        agg = None
        for frame in frames:
            mask = self.detector.detect_white_lines(frame)
            if agg is None:
                agg = mask.astype(np.uint8)
            else:
                agg = cv2.max(agg, mask.astype(np.uint8))

        if agg is None:
            return []

        # Projections
        col_sum = agg.sum(axis=0).astype(np.float32)
        row_sum = agg.sum(axis=1).astype(np.float32)

        # Smooth projections
        col_sum = cv2.blur(col_sum.reshape(1, -1), (1, 41)).flatten()
        row_sum = cv2.blur(row_sum.reshape(-1, 1), (41, 1)).flatten()

        def _find_peaks(arr, thresh_factor=1.0):
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            thresh = mean + std * thresh_factor
            peaks = np.where(arr > thresh)[0]
            if len(peaks) == 0:
                return []
            # group contiguous
            groups = []
            start = peaks[0]
            prev = peaks[0]
            for p in peaks[1:]:
                if p == prev + 1:
                    prev = p
                else:
                    groups.append((start, prev))
                    start = p
                    prev = p
            groups.append((start, prev))
            centers = [int((a + b) // 2) for a, b in groups]
            return centers

        dividers = _find_peaks(col_sum, thresh_factor=0.8)
        row_lines = _find_peaks(row_sum, thresh_factor=0.8)

        if len(dividers) < 3 or len(row_lines) < 2:
            return []

        def _merge_close(points, min_dist):
            if not points:
                return []
            pts = sorted(points)
            merged = [pts[0]]
            for p in pts[1:]:
                if p - merged[-1] < min_dist:
                    merged[-1] = int((merged[-1] + p) // 2)
                else:
                    merged.append(p)
            return merged

        def _select_spaced(points, values, min_dist, max_points):
            if not points:
                return []
            # sort candidates by value descending
            cand = sorted(zip(points, values), key=lambda x: x[1], reverse=True)
            selected = []
            for p, _v in cand:
                if all(abs(p - s) >= min_dist for s in selected):
                    selected.append(p)
                    if len(selected) >= max_points:
                        break
            return sorted(selected)

        min_div_dist = max(12, int(est_slot_w * 0.85))
        min_row_dist = max(10, int(est_slot_h * 0.85))

        dividers = _merge_close(dividers, min_div_dist)
        row_lines = _merge_close(row_lines, min_row_dist)

        # Hard cap peaks based on estimated slot size
        max_div = max(6, int(w / max(20, est_slot_w)))
        max_rows = max(2, int(h / max(20, est_slot_h)))

        # Use original projection values to select spaced strongest peaks
        div_values = [col_sum[p] for p in dividers]
        row_values = [row_sum[p] for p in row_lines]
        dividers = _select_spaced(dividers, div_values, min_div_dist, max_div)
        row_lines = _select_spaced(row_lines, row_values, min_row_dist, max_rows)

        if len(dividers) > 80:
            dividers = dividers[::2]
        if len(row_lines) > 60:
            row_lines = row_lines[::2]

        # Create rows between horizontal lines (use midpoints with spacing sanity)
        rows = []
        if len(row_lines) >= 2:
            row_spacings = [row_lines[i + 1] - row_lines[i] for i in range(len(row_lines) - 1)]
            med_row_spacing = int(np.median(row_spacings)) if row_spacings else est_slot_h
            for i in range(len(row_lines) - 1):
                spacing = row_lines[i + 1] - row_lines[i]
                if spacing < med_row_spacing * 0.5 or spacing > med_row_spacing * 1.8:
                    continue
                y_mid = (row_lines[i] + row_lines[i + 1]) // 2
                rows.append(y_mid)
        else:
            rows = []

        slots = []
        slot_id = 0
        # Estimate slot height from row spacing
        if len(row_lines) >= 2:
            row_spacings = [row_lines[i + 1] - row_lines[i] for i in range(len(row_lines) - 1)]
            slot_h = int(np.median(row_spacings)) if row_spacings else est_slot_h
        else:
            slot_h = est_slot_h

        # Split dividers into blocks by large gaps (drive lanes)
        if len(dividers) >= 2:
            gaps = [dividers[i + 1] - dividers[i] for i in range(len(dividers) - 1)]
            med_gap = int(np.median(gaps)) if gaps else est_slot_w
            block_break = max(int(med_gap * 2.5), int(est_slot_w * 2.5))
            blocks = []
            start = 0
            for i, gap in enumerate(gaps):
                if gap >= block_break:
                    blocks.append(dividers[start : i + 1])
                    start = i + 1
            blocks.append(dividers[start:])
        else:
            blocks = [dividers]

        for row_idx, y_mid in enumerate(rows):
            y1 = max(0, int(y_mid - slot_h // 2))
            y2 = min(h, int(y_mid + slot_h // 2))
            for block in blocks:
                if len(block) < 2:
                    continue
                for i in range(len(block) - 1):
                    x1 = block[i]
                    x2 = block[i + 1]
                    width = x2 - x1
                    if width < int(est_slot_w * 0.5) or width > int(est_slot_w * 2.2):
                        continue
                    slots.append(
                        {
                            "id": slot_id,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "center": (int((x1 + x2) // 2), int((y1 + y2) // 2)),
                            "row": row_idx,
                            "occupied": False,
                            "confidence": 0.0,
                            "vehicle": None,
                        }
                    )
                    slot_id += 1

        return slots

    def _slots_from_vehicle_gaps(self, frames: List[np.ndarray]) -> List[Dict]:
        if not frames:
            return []

        img_h, img_w = frames[0].shape[:2]
        vehicles_all = []
        widths = []
        heights = []
        for frame in frames:
            vehicles = self.detector.detect_vehicles(frame)
            if len(vehicles) < 2:
                vehicles = self._detect_vehicles_with_conf(frame, conf=0.01)
            for v in vehicles:
                vehicles_all.append(v)
                widths.append(v["bbox"][2] - v["bbox"][0])
                heights.append(v["bbox"][3] - v["bbox"][1])

        if len(vehicles_all) < 2:
            return []

        med_w = int(np.median(widths))
        med_h = int(np.median(heights))
        slot_w_base = int(med_w * 1.25)
        slot_h = int(med_h * 1.25)

        # Cluster vehicles into rows by Y position
        from sklearn.cluster import DBSCAN

        y_centers = np.array([[v["center"][1]] for v in vehicles_all])
        row_eps = max(10, int(med_h * 0.8))
        clustering = DBSCAN(eps=row_eps, min_samples=1).fit(y_centers)

        rows_dict = {}
        for idx, label in enumerate(clustering.labels_):
            rows_dict.setdefault(label, []).append(vehicles_all[idx])

        rows = list(rows_dict.values())
        rows.sort(key=lambda r: np.mean([v["center"][1] for v in r]))

        slots = []
        slot_id = 0
        for row_idx, row_vehicles in enumerate(rows):
            centers_x = sorted([v["center"][0] for v in row_vehicles])
            if len(centers_x) >= 2:
                spacings = [centers_x[i + 1] - centers_x[i] for i in range(len(centers_x) - 1)]
                med_spacing = int(np.median(spacings))
                slot_w = max(int(slot_w_base * 0.7), min(int(slot_w_base * 1.6), med_spacing))
            else:
                slot_w = slot_w_base

            row_y = int(np.median([v["center"][1] for v in row_vehicles]))
            y1 = max(0, row_y - slot_h // 2)
            y2 = min(img_h, row_y + slot_h // 2)

            min_x = min([v["bbox"][0] for v in row_vehicles])
            max_x = max([v["bbox"][2] for v in row_vehicles])

            row_start = max(0, min_x - slot_w * 3)
            row_end = min(img_w, max_x + slot_w * 3)

            x = row_start
            while x + slot_w <= row_end:
                slots.append(
                    {
                        "id": slot_id,
                        "bbox": (int(x), int(y1), int(x + slot_w), int(y2)),
                        "center": (int(x + slot_w // 2), int((y1 + y2) // 2)),
                        "row": row_idx,
                        "occupied": False,
                        "confidence": 0.0,
                        "vehicle": None,
                    }
                )
                slot_id += 1
                x += slot_w

        return slots

    def _detect_vehicles_with_conf(self, image: np.ndarray, conf: float) -> List[Dict]:
        """Lower-confidence vehicle detection for sparse frames."""
        results = self.detector.model(image, conf=conf, verbose=False)
        vehicles = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) in self.detector.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_v = float(box.conf[0])
                    vehicles.append({
                        "bbox": (x1, y1, x2, y2),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "confidence": conf_v,
                    })
        return vehicles

    def _assign_occupancy(self, slots: List[Dict], vehicles: List[Dict]) -> List[Dict]:
        for slot in slots:
            sx1, sy1, sx2, sy2 = slot["bbox"]
            slot_area = max(1, (sx2 - sx1) * (sy2 - sy1))
            best_overlap = 0.0
            best_vehicle = None

            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle["bbox"]
                ix1 = max(sx1, vx1)
                iy1 = max(sy1, vy1)
                ix2 = min(sx2, vx2)
                iy2 = min(sy2, vy2)
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    overlap_ratio = intersection / slot_area
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_vehicle = vehicle

                vcx, vcy = vehicle["center"]
                if sx1 <= vcx <= sx2 and sy1 <= vcy <= sy2:
                    best_overlap = max(best_overlap, 0.5)
                    best_vehicle = vehicle

            if best_overlap > self.overlap_threshold:
                slot["occupied"] = True
                slot["confidence"] = min(0.90 + best_overlap * 0.1, 0.99)
                slot["vehicle"] = best_vehicle
            else:
                slot["occupied"] = False
                slot["confidence"] = 0.95
                slot["vehicle"] = None

        return slots

    def _apply_smoothing(self, slots: List[Dict]) -> List[Dict]:
        for slot in slots:
            sid = slot["id"]
            raw = 1 if slot["occupied"] else 0

            state = self._slot_state.get(sid)
            if state is None:
                self._slot_state[sid] = {"state": raw, "streak": 0}
                continue

            if raw == state["state"]:
                state["streak"] = 0
            else:
                state["streak"] += 1
                if state["streak"] >= self.confirm_frames:
                    state["state"] = raw
                    state["streak"] = 0

            slot["occupied"] = bool(state["state"])

        return slots

    @staticmethod
    def _scale_slots_if_needed(slots, from_shape, to_shape):
        if from_shape == to_shape:
            return slots

        from_h, from_w = from_shape
        to_h, to_w = to_shape
        sx = to_w / max(1, from_w)
        sy = to_h / max(1, from_h)

        scaled = []
        for s in slots:
            x1, y1, x2, y2 = s["bbox"]
            scaled.append(
                {
                    **s,
                    "bbox": (
                        int(x1 * sx),
                        int(y1 * sy),
                        int(x2 * sx),
                        int(y2 * sy),
                    ),
                    "center": (
                        int(s["center"][0] * sx),
                        int(s["center"][1] * sy),
                    ),
                }
            )
        return scaled

    @staticmethod
    def _reindex_slots(slots: List[Dict]) -> List[Dict]:
        for idx, s in enumerate(slots):
            s["id"] = idx
        return slots

    @staticmethod
    def _slot_geometry_variation(slots: List[Dict]) -> float:
        if not slots:
            return 1.0

        widths = [max(1, s["bbox"][2] - s["bbox"][0]) for s in slots]
        heights = [max(1, s["bbox"][3] - s["bbox"][1]) for s in slots]

        w_mean = float(np.mean(widths))
        h_mean = float(np.mean(heights))
        w_cv = float(np.std(widths)) / max(1.0, w_mean)
        h_cv = float(np.std(heights)) / max(1.0, h_mean)

        return w_cv + h_cv


def _load_calibration_frames(video_path: str, max_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smart Parking auto-calibration + live processing")
    parser.add_argument("--video", default="CarPark.mp4", help="Path to input video")
    parser.add_argument("--calib-frames", type=int, default=20, help="Frames to use for calibration")
    parser.add_argument("--slots", default="slots_auto.json", help="Path to save/load slots")
    parser.add_argument("--no-calib", action="store_true", help="Skip calibration and just load slots")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Frames required to flip occupancy")
    parser.add_argument("--min-slots", type=int, default=10, help="Minimum slots for calibration candidate")
    parser.add_argument("--max-slots", type=int, default=60, help="Maximum slots for calibration candidate")
    parser.add_argument(
        "--calib-mode",
        default="auto",
        choices=["auto", "vehicle", "lines", "projection", "vehicle_gaps"],
        help="Calibration mode: auto, vehicle-grid, line-dividers, projection, or vehicle-gaps.",
    )

    args = parser.parse_args()

    pipeline = ParkingSlotPipeline(confirm_frames=args.confirm_frames)

    if not args.no_calib:
        frames = _load_calibration_frames(args.video, args.calib_frames)
        if not frames:
            raise RuntimeError(f"No frames read from {args.video}")
        pipeline.calibrate_from_frames(
            frames,
            save_path=args.slots,
            min_slots=args.min_slots,
            max_slots=args.max_slots,
            mode=args.calib_mode,
        )

    pipeline.load_slots(args.slots)

    cap = cv2.VideoCapture(args.video)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.process_frame(frame)
        annotated = pipeline.annotate_frame(frame, result["slots"])
        cv2.imshow("Smart Parking", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
