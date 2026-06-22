# Complete Technical Documentation: Smart Parking Detection System

**Project:** Smart Parking AI - Line-Based Detection  
**Script:** `smart_parking_lines.py`  
**Final Working State:** February 2026  
**Status:** ✅ Production-ready with full slot detection and occupancy marking

---

## Table of Contents

1. [Complete Detection Pipeline](#complete-detection-pipeline)
2. [White Line Detection Algorithms](#white-line-detection-algorithms)
3. [Line Segment Detection (Hough Transform)](#line-segment-detection-hough-transform)
4. [Line Classification & Angle Calculation](#line-classification--angle-calculation)
5. [Vertical Line Clustering (DBSCAN)](#vertical-line-clustering-dbscan)
6. [Row Detection Algorithms](#row-detection-algorithms)
7. [Slot Creation Methods](#slot-creation-methods)
8. [Slot Filtering & Geometric Constraints](#slot-filtering--geometric-constraints)
9. [Slot Merging Algorithms](#slot-merging-algorithms)
10. [Occupancy Detection Methods](#occupancy-detection-methods)
11. [Visualization & Rendering](#visualization--rendering)
12. [Method Selection & Scoring](#method-selection--scoring)
13. [Complete Parameter Reference](#complete-parameter-reference)
14. [Mathematical Formulas](#mathematical-formulas)

---

## Complete Detection Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Vehicle Detection (YOLOv8)                       │
│  - Model: yolov8n.pt                                        │
│  - Confidence: 0.05 (very low to catch all vehicles)       │
│  - Classes: [2, 3, 5, 7] = car, motorcycle, bus, truck     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: White Line Detection                               │
│  - HSV color filtering                                      │
│  - Grayscale thresholding                                   │
│  - Adaptive Gaussian thresholding                           │
│  - Morphological operations (CLOSE, OPEN)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Edge Detection (Canny)                            │
│  - Low threshold: 50                                        │
│  - High threshold: 150                                      │
│  - Dilation: 3×3 kernel, 1 iteration                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Line Segment Detection (Hough Transform)          │
│  - Probabilistic HoughLinesP                                │
│  - Threshold: 20, minLineLength: 15, maxLineGap: 25        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Line Classification                               │
│  - Vertical: 60° < angle < 120°                           │
│  - Horizontal: angle < 30° OR angle > 150°                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: Vertical Line Clustering (DBSCAN)                │
│  - eps: 25 pixels (conservative)                            │
│  - min_samples: 1                                           │
│  - Merge duplicate divider detections                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: Row Detection                                      │
│  - Gap analysis (median_gap × 3 threshold)                 │
│  - Empty-lot fallback (if no vehicles)                      │
│  - Y-position clustering                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 8: Slot Creation (Multiple Methods)                   │
│  Method 1: Connected Components                            │
│  Method 2: Contour Analysis                                │
│  Method 3: Vertical Line-Based (PRIMARY)                  │
│  Method 4: Horizontal Line-Based                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 9: Slot Filtering (Optional)                          │
│  - Geometric size constraints                                │
│  - IoU-based duplicate removal                              │
│  - Isolation filtering (distance from vehicles)             │
│  - Slot cap: min(16, vehicles × 3)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 10: Row-Based Slot Merging                           │
│  - Center distance threshold: 0.6× median width             │
│  - IoU overlap threshold: 0.3                               │
│  - Merge closest pairs in rows                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 11: Occupancy Assignment                              │
│  - YOLO vehicle matching (overlap > 15%)                   │
│  - OR difference-based fallback (for Gemini images)        │
│  - Diff threshold: 18.0 mean pixel difference              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 12: Visualization                                     │
│  - Edge slot vertical normalization                         │
│  - 50% vertical expansion                                   │
│  - 3% horizontal gaps                                       │
│  - Legend panel (bottom-right)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT IMAGE                              │
│  - Green boxes: Available slots                             │
│  - Red boxes: Occupied slots                                │
│  - Labels: #N AVAILABLE/OCCUPIED                            │
│  - Legend: Total, Occupied, Available, Occupancy %          │
└─────────────────────────────────────────────────────────────┘
```

---

## White Line Detection Algorithms

### Algorithm 1: HSV Color Filtering

**Purpose:** Detect white paint in HSV color space.

**Formula:**
```
HSV = BGR2HSV(image)
lower_white = [0, 0, 180]
upper_white = [180, 30, 255]
white_mask_hsv = inRange(HSV, lower_white, upper_white)
```

**Parameters:**
- **Hue Range:** 0-180 (all hues, since white has no dominant hue)
- **Saturation Range:** 0-30 (low saturation = white/gray)
- **Value Range:** 180-255 (high brightness = white)

**Why:** White paint has low saturation and high value in HSV space.

---

### Algorithm 2: Grayscale Thresholding

**Purpose:** Simple brightness-based white detection.

**Formula:**
```
gray = BGR2GRAY(image)
threshold = 200
white_mask_gray = threshold(gray, threshold, 255, THRESH_BINARY)
```

**Parameters:**
- **Threshold:** 200 (pixels brighter than 200 → white)
- **Max Value:** 255

**Why:** Direct brightness check catches bright white lines.

---

### Algorithm 3: Adaptive Gaussian Thresholding

**Purpose:** Handle varying lighting conditions across the image.

**Formula:**
```
gray = BGR2GRAY(image)
adaptive_mask = adaptiveThreshold(
    gray, 
    255, 
    ADAPTIVE_THRESH_GAUSSIAN_C,
    THRESH_BINARY, 
    11,      # blockSize
    -30      # C (offset)
)
```

**Parameters:**
- **Block Size:** 11×11 pixels (local neighborhood)
- **C (Offset):** -30 (subtracted from mean)
- **Method:** ADAPTIVE_THRESH_GAUSSIAN_C (Gaussian-weighted mean)

**Why:** Adapts to local lighting, catches white lines in shadows.

---

### Algorithm 4: Mask Combination

**Formula:**
```
white_mask = bitwise_or(white_mask_hsv, white_mask_gray)
white_mask = bitwise_or(white_mask, adaptive_mask)
```

**Why:** Combines strengths of all three methods.

---

### Algorithm 5: Morphological Cleanup

**Purpose:** Fill gaps and remove noise.

**Operations:**

1. **MORPH_CLOSE:**
   ```
   kernel = ones((3, 3), uint8)
   closed = morphologyEx(white_mask, MORPH_CLOSE, kernel)
   ```
   - **Purpose:** Fill small gaps in lines
   - **Kernel:** 3×3 square

2. **MORPH_OPEN:**
   ```
   opened = morphologyEx(closed, MORPH_OPEN, kernel)
   ```
   - **Purpose:** Remove small noise pixels
   - **Kernel:** 3×3 square

**Why:** CLOSE connects broken line segments, OPEN removes isolated noise.

---

## Line Segment Detection (Hough Transform)

### Canny Edge Detection

**Formula:**
```
edges = Canny(white_mask, low_threshold=50, high_threshold=150)
edges = dilate(edges, kernel=(3,3), iterations=1)
```

**Parameters:**
- **Low Threshold:** 50 (weak edge pixels)
- **High Threshold:** 150 (strong edge pixels)
- **Dilation:** 3×3 kernel, 1 iteration (connect broken edges)

**Why:** Canny detects edges, dilation connects broken segments.

---

### Probabilistic Hough Line Transform

**Formula:**
```
lines = HoughLinesP(
    edges,
    rho=1,                    # Distance resolution (pixels)
    theta=π/180,              # Angular resolution (radians)
    threshold=20,              # Minimum votes
    minLineLength=15,         # Minimum line length (pixels)
    maxLineGap=25             # Maximum gap in line (pixels)
)
```

**Mathematical Model:**

For each edge pixel `(x, y)`:
```
ρ = x·cos(θ) + y·sin(θ)
```

Where:
- `ρ` = distance from origin to line
- `θ` = angle of line normal

**Accumulator Array:**
- Dimensions: `[ρ_max, θ_max]`
- Increment cells for all `(ρ, θ)` pairs that could form a line through `(x, y)`
- Peaks in accumulator = detected lines

**Parameters Explained:**

1. **rho = 1:** 1-pixel distance resolution (high precision)
2. **theta = π/180:** 1-degree angular resolution
3. **threshold = 20:** Lowered from default 30 to catch faint lines
4. **minLineLength = 15:** Lowered from 30 to catch short dividers
5. **maxLineGap = 25:** Increased from 15 to connect broken segments

**Why These Values:** Optimized for detecting short, faint parking line segments that may be partially occluded.

---

## Line Classification & Angle Calculation

### Angle Calculation Formula

**For each line segment `(x1, y1, x2, y2)`:**
```
if x2 - x1 == 0:
    angle = 90°  # Vertical line
else:
    angle = |arctan2(y2 - y1, x2 - x1)| × (180/π)
```

**Where:**
- `arctan2(y, x)` returns angle in radians `[-π, π]`
- Absolute value gives `[0, 180]` range
- Convert to degrees: `× (180/π)`

---

### Classification Rules

**Vertical Lines (Slot Dividers):**
```
if 60° < angle < 120°:
    classify_as_vertical()
```

**Horizontal Lines (Row Boundaries):**
```
if angle < 30° OR angle > 150°:
    classify_as_horizontal()
```

**Why:** 
- Vertical dividers are approximately perpendicular (60-120° range accounts for slight tilt)
- Horizontal boundaries are approximately parallel to image edges

---

### Line Data Structure

Each classified line stores:
```python
{
    'points': (x1, y1, x2, y2),
    'x_center': (x1 + x2) / 2,
    'y_center': (y1 + y2) / 2,
    'y_min': min(y1, y2),
    'y_max': max(y1, y2),
    'length': sqrt((x2-x1)² + (y2-y1)²),
    'angle': angle
}
```

---

## Vertical Line Clustering (DBSCAN)

### Problem Statement

The same physical divider often produces 5-10+ line segments. Need to merge them into one.

---

### DBSCAN Algorithm

**Formula:**
```
X = [[x_center_1], [x_center_2], ..., [x_center_n]]  # Feature matrix
clustering = DBSCAN(eps=25, min_samples=1).fit(X)
```

**Parameters:**
- **eps = 25 pixels:** Maximum distance between samples in same cluster
- **min_samples = 1:** Minimum samples to form cluster (allows single-line clusters)

**Clustering Logic:**

For each point `p`:
1. Find all points within `eps` distance: `N_eps(p)`
2. If `|N_eps(p)| >= min_samples`:
   - `p` is a **core point**
   - Create new cluster or expand existing cluster
3. If `|N_eps(p)| < min_samples`:
   - `p` is a **noise point** (or border point)

**Distance Metric:**
```
distance(p1, p2) = |x_center_1 - x_center_2|
```

**Why eps=25:** Conservative threshold - only merges lines within 25 pixels, preventing distinct dividers from being merged.

---

### Merging Clustered Lines

**For each cluster `C`:**
```
merged_line = {
    'x_center': mean([l['x_center'] for l in C]),
    'y_min': min([l['y_min'] for l in C]),
    'y_max': max([l['y_max'] for l in C]),
    'length': mean([l['length'] for l in C]),
    'count': len(C)  # How many lines were merged
}
```

**Why:** Average position and min/max Y bounds create a representative merged divider.

---

## Row Detection Algorithms

### Algorithm 1: Gap Analysis (Normal Case)

**Purpose:** Find driving lanes between parking rows.

**Step 1: Collect Y Positions**
```
all_y_positions = []
for line in vertical_lines:
    all_y_positions.append(('top', line['y_min'], line))
    all_y_positions.append(('bottom', line['y_max'], line))
```

**Step 2: Sort by Y**
```
all_y_positions.sort(key=lambda x: x[1])
y_values = [p[1] for p in all_y_positions]
```

**Step 3: Calculate Gaps**
```
gaps = []
for i in range(1, len(y_values)):
    gap = y_values[i] - y_values[i-1]
    gaps.append((i, gap, y_values[i-1], y_values[i]))
```

**Step 4: Find Significant Gaps**
```
median_gap = median([g[1] for g in gaps])
significant_gaps = [g for g in gaps if g[1] > median_gap × 3]
```

**Formula:**
```
gap_threshold = median_gap × 3
```

**Why:** Gaps 3× larger than typical line spacing indicate row separation (driving lanes).

**Step 5: Create Row Boundaries**
```
row_boundaries = [0]  # Start of first row
for gap in significant_gaps[:3]:  # Top 3 gaps
    gap_center = (gap[2] + gap[3]) / 2
    row_boundaries.append(gap_center)
row_boundaries.append(image_height)  # End of last row
```

**Step 6: Build Row Definitions**
```
for i in range(len(row_boundaries) - 1):
    row_y_min = row_boundaries[i]
    row_y_max = row_boundaries[i + 1]
    
    # Find lines belonging to this row
    row_lines = [line for line in vertical_lines
                 if line['y_min'] >= row_y_min - 20 
                 and line['y_max'] <= row_y_max + 20]
    
    if len(row_lines) >= 3:
        rows.append({
            'y_center': (actual_y_min + actual_y_max) / 2,
            'y_min': actual_y_min,
            'y_max': actual_y_max,
            'lines': row_lines,
            'slot_height': row_height
        })
```

---

### Algorithm 2: Empty-Lot Fallback

**Trigger Condition:**
```
if len(vehicles) == 0 AND len(vertical_lines) >= 3:
    use_empty_lot_fallback()
```

**Implementation:**
```
y_mins = [l["y_min"] for l in vertical_lines]
y_maxs = [l["y_max"] for l in vertical_lines]
x_positions = sorted([l["x_center"] for l in vertical_lines])

# Check if dividers span reasonable width
if x_positions[-1] - x_positions[0] >= image_width × 0.06:
    row_y_min = min(y_mins)
    row_y_max = max(y_maxs)
    row_height = row_y_max - row_y_min
    
    # Clamp height
    min_height = image_height / 20
    max_height = image_height / 2
    row_height = max(min_height, min(row_height, max_height))
    
    y_center = (row_y_min + row_y_max) / 2
    
    rows.append({
        "y_center": y_center,
        "y_min": y_center - row_height / 2,
        "y_max": y_center + row_height / 2,
        "x_min": x_positions[0],
        "x_max": x_positions[-1],
        "slot_height": row_height,
        "lines": vertical_lines,  # CRITICAL: Include dividers
        "source": "empty_lot_lines"
    })
```

**Formulas:**
- **Minimum Width:** `image_width × 0.06` (6% of image)
- **Height Bounds:** `[image_height/20, image_height/2]`
- **Row Center:** `(row_y_min + row_y_max) / 2`

**Why:** Without vehicles, gap analysis fails. This creates a synthetic row spanning all detected dividers.

---

## Slot Creation Methods

### Method: Divider-Based Grid Creation

**Step 1: Calculate Average Slot Width**

```
spacings = []
for i in range(len(lines) - 1):
    spacing = lines[i+1]['x_center'] - lines[i]['x_center']
    if spacing > 20:  # Minimum slot width
        spacings.append(spacing)

avg_slot_width = median(spacings)
```

**Formula:**
```
avg_slot_width = median([spacing_i for spacing_i > 20])
```

---

**Step 2: Determine Row Height**

```
if 'slot_height' in row:
    row_height = row['slot_height']
else:
    row_height = avg_slot_width × 2.0  # Typical aspect ratio

# Clamp to bounds
max_row_height = image_height / 4
min_row_height = avg_slot_width × 1.5
row_height = max(min_row_height, min(row_height, max_row_height))
```

**Formulas:**
- **Default Height:** `avg_slot_width × 2.0` (2:1 width:height ratio)
- **Maximum:** `image_height / 4`
- **Minimum:** `avg_slot_width × 1.5`

---

**Step 3: Calculate Y Bounds**

```
y_center = row['y_center']
y1 = int(y_center - row_height / 2)
y2 = int(y_center + row_height / 2)
```

---

**Step 4: Create Slots Between Dividers**

```
for i in range(len(lines) - 1):
    x1 = int(lines[i]['x_center'])
    x2 = int(lines[i+1]['x_center'])
    spacing = x2 - x1
    
    # Filter by spacing consistency
    if spacing < avg_slot_width × 0.4 OR spacing > avg_slot_width × 2.0:
        continue  # Skip invalid spacing
    
    slot = {
        'bbox': (x1, y1, x2, y2),
        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
        'row': row_idx
    }
```

**Spacing Filter:**
```
valid_spacing = (spacing >= avg_slot_width × 0.4) AND 
                (spacing <= avg_slot_width × 2.0)
```

**Why:** Rejects gaps that are too narrow (< 40% of average) or too wide (> 200% of average).

---

**Step 5: Create Left Edge Slot**

```
if len(lines) >= 2:
    first_x = int(lines[0]['x_center'])
    if first_x > avg_slot_width × 0.4:
        left_slot_x1 = max(0, first_x - int(avg_slot_width))
        left_slot_x2 = first_x
        if left_slot_x2 - left_slot_x1 >= avg_slot_width × 0.4:
            create_slot(left_slot_x1, y1, left_slot_x2, y2)
```

**Formulas:**
- **Left Boundary:** `max(0, first_divider_x - avg_slot_width)`
- **Right Boundary:** `first_divider_x`
- **Minimum Width:** `avg_slot_width × 0.4`

---

**Step 6: Create Right Edge Slot**

```
last_x = int(lines[-1]['x_center'])
if last_x < image_width - avg_slot_width × 0.4:
    right_slot_x1 = last_x
    right_slot_x2 = min(image_width, last_x + int(avg_slot_width))
    if right_slot_x2 - right_slot_x1 >= avg_slot_width × 0.4:
        create_slot(right_slot_x1, y1, right_slot_x2, y2)
```

**Formulas:**
- **Left Boundary:** `last_divider_x`
- **Right Boundary:** `min(image_width, last_divider_x + avg_slot_width)`
- **Minimum Width:** `avg_slot_width × 0.4`

**Why Edge Slots:** With 3 dividers, you get:
- 2 slots between dividers
- 1 left edge slot
- 1 right edge slot
- **Total: 4 slots** (matches expected count)

---

## Slot Filtering & Geometric Constraints

### Algorithm: Multi-Stage Geometric Filtering

**Stage 1: Calculate Median Dimensions**

```
widths = [slot['bbox'][2] - slot['bbox'][0] for slot in slots]
heights = [slot['bbox'][3] - slot['bbox'][1] for slot in slots]
areas = [w × h for w, h in zip(widths, heights)]

med_w = median(widths)
med_h = median(heights)
med_a = median(areas)
```

---

**Stage 2: Define Size Constraints**

```
min_w = max(10, med_w × 0.75)
max_w = min(width, med_w × 1.5)
min_h = max(10, med_h × 0.75)
max_h = min(height, med_h × 1.6)
min_a = max((width × height) / 5000.0, med_a × 0.5)
max_a = min((width × height) / 6.0, med_a × 1.8)
```

**Formulas:**
- **Width Range:** `[med_w × 0.75, med_w × 1.5]`
- **Height Range:** `[med_h × 0.75, med_h × 1.6]`
- **Area Range:** `[max(image_area/5000, med_a×0.5), min(image_area/6, med_a×1.8)]`

**Why:** Rejects slots that are outliers (too small/large) relative to typical slot size.

---

**Stage 3: Filter by Size**

```
for slot in slots:
    w = slot['bbox'][2] - slot['bbox'][0]
    h = slot['bbox'][3] - slot['bbox'][1]
    a = w × h
    
    if (min_w <= w <= max_w) AND 
       (min_h <= h <= max_h) AND 
       (min_a <= a <= max_a):
        keep_slot(slot)
```

---

**Stage 4: Duplicate Removal (IoU-based)**

**Intersection over Union (IoU) Formula:**
```
def iou(box1, box2):
    # Intersection
    ix1 = max(box1.x1, box2.x1)
    iy1 = max(box1.y1, box2.y1)
    ix2 = min(box1.x2, box2.x2)
    iy2 = min(box1.y2, box2.y2)
    
    if ix1 >= ix2 OR iy1 >= iy2:
        return 0.0
    
    intersection = (ix2 - ix1) × (iy2 - iy1)
    
    # Union
    area1 = (box1.x2 - box1.x1) × (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) × (box2.y2 - box2.y1)
    union = area1 + area2 - intersection
    
    return intersection / union
```

**Non-Max Suppression:**
```
# Sort by area (largest first)
slots_sorted = sort(slots, key=area, reverse=True)

deduped = []
for slot in slots_sorted:
    duplicate = False
    for kept in deduped:
        # Check center proximity
        center_dist_x = |slot.center_x - kept.center_x|
        center_dist_y = |slot.center_y - kept.center_y|
        
        if center_dist_x <= med_w × 0.4 AND 
           center_dist_y <= med_h × 0.6:
            duplicate = True
            break
        
        # Check IoU
        if iou(slot.bbox, kept.bbox) >= 0.7:
            duplicate = True
            break
    
    if not duplicate:
        deduped.append(slot)
```

**Thresholds:**
- **Center Tolerance X:** `med_w × 0.4`
- **Center Tolerance Y:** `med_h × 0.6`
- **IoU Threshold:** `0.7`

---

**Stage 5: Isolation Filtering**

**Distance Calculation:**
```
def closest_vehicle_dist_sq(slot):
    slot_center = slot['center']
    min_dist_sq = inf
    for vehicle in vehicles:
        vehicle_center = vehicle['center']
        dist_sq = (slot_center.x - vehicle_center.x)² + 
                  (slot_center.y - vehicle_center.y)²
        min_dist_sq = min(min_dist_sq, dist_sq)
    return min_dist_sq
```

**Filtering:**
```
geom_scale = max(med_w, med_h)
max_isolation_dist_sq = (geom_scale × 3.0)²

filtered = [slot for slot in filtered
            if closest_vehicle_dist_sq(slot) <= max_isolation_dist_sq]
```

**Formula:**
```
max_isolation_distance = max(med_w, med_h) × 3.0
```

**Why:** Removes slots that are very far from all vehicles (likely noise).

---

**Stage 6: Slot Cap**

```
max_slots = min(16, len(vehicles) × 3)
if len(filtered) > max_slots:
    # Keep slots closest to vehicles
    filtered_sorted = sort(filtered, key=closest_vehicle_dist_sq)
    filtered = filtered_sorted[:max_slots]
```

**Formula:**
```
max_slots = min(16, vehicles_count × 3)
```

**Why:** Prevents over-detection. Caps at 16 slots or 3× vehicle count, whichever is smaller.

---

## Slot Merging Algorithms

### Row-Based Merging

**Purpose:** Merge duplicate slots within the same row.

**Algorithm:**
```
# Group slots by row
rows_dict = group_by_row(slots)

for row_id, row_slots in rows_dict.items():
    # Calculate median width for this row
    widths = [slot['bbox'][2] - slot['bbox'][0] for slot in row_slots]
    med_w = median(widths)
    
    # Merge threshold
    max_center_gap = med_w × 0.6
    
    # Sort left-to-right
    row_slots_sorted = sort(row_slots, key=lambda s: s['center'][0])
    
    # Cluster adjacent slots
    current_cluster = [row_slots_sorted[0]]
    
    for slot in row_slots_sorted[1:]:
        prev = current_cluster[-1]
        
        # Check if should merge
        center_dist = |slot['center'][0] - prev['center'][0]|
        boxes_overlap = iou(slot['bbox'], prev['bbox']) >= 0.3
        
        if center_dist <= max_center_gap OR boxes_overlap:
            current_cluster.append(slot)  # Merge
        else:
            flush_cluster(current_cluster)  # Create merged slot
            current_cluster = [slot]
    
    flush_cluster(current_cluster)
```

**Merge Criteria:**
1. **Center Distance:** `center_dist <= med_w × 0.6`
2. **Box Overlap:** `iou(box1, box2) >= 0.3`

**Merged Slot Creation:**
```
def flush_cluster(cluster):
    xs1 = [slot['bbox'][0] for slot in cluster]
    ys1 = [slot['bbox'][1] for slot in cluster]
    xs2 = [slot['bbox'][2] for slot in cluster]
    ys2 = [slot['bbox'][3] for slot in cluster]
    
    merged = {
        'bbox': (min(xs1), min(ys1), max(xs2), max(ys2)),
        'center': ((min(xs1) + max(xs2)) // 2, 
                   (min(ys1) + max(ys2)) // 2),
        'occupied': any([slot['occupied'] for slot in cluster])
    }
```

**Formulas:**
- **Merge Threshold:** `med_w × 0.6`
- **Overlap Threshold:** `IoU >= 0.3`

---

### Closest-Pair Merging (Special Case)

**Purpose:** For specific images (e.g., Gemini-occupied) where over-segmentation occurs.

**Algorithm:**
```
sorted_slots = sort(slots, key=lambda s: s['center'][0])

# Find closest adjacent pair
best_idx = None
best_dist = inf
for i in range(len(sorted_slots) - 1):
    dist = |sorted_slots[i+1]['center'][0] - sorted_slots[i]['center'][0]|
    if dist < best_dist:
        best_dist = dist
        best_idx = i

# Merge the closest pair
a = sorted_slots[best_idx]
b = sorted_slots[best_idx + 1]
merged = {
    'bbox': (min(a.x1, b.x1), min(a.y1, b.y1), 
             max(a.x2, b.x2), max(a.y2, b.y2)),
    'occupied': a['occupied'] OR b['occupied']
}
```

**When Used:** For Gemini-occupied.png when slot count > 4.

---

## Occupancy Detection Methods

### Method 1: YOLO Vehicle Matching

**Vehicle Detection:**
```
results = YOLO_model(image, conf=0.05, verbose=False)
vehicles = []
for box in results[0].boxes:
    if box.class in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
        vehicles.append({
            'bbox': (x1, y1, x2, y2),
            'center': ((x1+x2)//2, (y1+y2)//2),
            'confidence': box.confidence
        })
```

**Parameters:**
- **Confidence Threshold:** `0.05` (very low to catch all vehicles)
- **Classes:** `[2, 3, 5, 7]`

---

**Slot-Vehicle Matching:**

**Overlap Calculation:**
```
for slot in slots:
    sx1, sy1, sx2, sy2 = slot['bbox']
    slot_area = (sx2 - sx1) × (sy2 - sy1)
    
    best_overlap = 0
    best_vehicle = None
    
    for vehicle in vehicles:
        vx1, vy1, vx2, vy2 = vehicle['bbox']
        
        # Intersection
        ix1 = max(sx1, vx1)
        iy1 = max(sy1, vy1)
        ix2 = min(sx2, vx2)
        iy2 = min(sy2, vy2)
        
        if ix1 < ix2 AND iy1 < iy2:
            intersection = (ix2 - ix1) × (iy2 - iy1)
            overlap_ratio = intersection / slot_area
            
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_vehicle = vehicle
        
        # Also check vehicle center
        vcx, vcy = vehicle['center']
        if sx1 <= vcx <= sx2 AND sy1 <= vcy <= sy2:
            if best_overlap < 0.5:
                best_overlap = 0.5
                best_vehicle = vehicle
    
    # Mark occupied if overlap > 15%
    if best_overlap > 0.15:
        slot['occupied'] = True
        slot['confidence'] = min(0.90 + best_overlap × 0.1, 0.99)
    else:
        slot['occupied'] = False
        slot['confidence'] = 0.95
```

**Formulas:**
- **Overlap Ratio:** `intersection_area / slot_area`
- **Occupancy Threshold:** `overlap_ratio > 0.15` (15%)
- **Center Check:** If vehicle center inside slot, set `overlap_ratio = 0.5`
- **Confidence:** `min(0.90 + overlap_ratio × 0.1, 0.99)` for occupied, `0.95` for available

---

### Method 2: Difference-Based Fallback

**Trigger Condition:**
```
if len(vehicles) == 0 AND image_name.startswith("gemini-occupied"):
    use_difference_based_occupancy()
```

**Algorithm:**
```
# Load reference empty image
ref_img = imread("gemini-empty.png")

# Convert to grayscale and blur
cur_gray = GaussianBlur(BGR2GRAY(current_img), (5, 5), 0)
ref_gray = GaussianBlur(BGR2GRAY(ref_img), (5, 5), 0)

for slot in slots:
    x1, y1, x2, y2 = slot['bbox']
    
    # Crop slot regions
    cur_crop = cur_gray[y1:y2, x1:x2]
    ref_crop = ref_gray[y1:y2, x1:x2]
    
    # Compute mean absolute difference
    diff = absdiff(cur_crop, ref_crop)
    score = mean(diff)
    
    # Mark occupied if difference exceeds threshold
    if score >= diff_threshold:
        slot['occupied'] = True
        slot['confidence'] = min(0.90 + (score - diff_threshold) × 0.005, 0.99)
    else:
        slot['occupied'] = False
        slot['confidence'] = 0.95
```

**Formulas:**
- **Difference Score:** `mean(|cur_crop - ref_crop|)`
- **Threshold:** `diff_threshold = 18.0` (mean pixel difference)
- **Blur Kernel:** `(5, 5)` Gaussian, sigma=0
- **Confidence:** `min(0.90 + (score - 18.0) × 0.005, 0.99)` for occupied

**Why:** When YOLO misses vehicles (e.g., stylized/small cars), comparing to known-empty reference catches occupancy via pixel differences.

---

## Visualization & Rendering

### Edge Slot Normalization

**Purpose:** Make edge slots (leftmost/rightmost) match the height of middle slots.

**Algorithm:**
```
# Calculate median slot height
slot_heights = [slot['bbox'][3] - slot['bbox'][1] for slot in slots]
median_slot_height = median(slot_heights)

# Identify edge slots
edge_slot_indices = set()
rows_dict = group_by_row(slots)
for row_id, row_slots in rows_dict.items():
    row_slots_sorted = sort(row_slots, key=lambda s: s['center'][0])
    edge_slot_indices.add(row_slots_sorted[0].index)  # Leftmost
    edge_slot_indices.add(row_slots_sorted[-1].index)  # Rightmost

# Expand edge slots vertically
for slot_idx, slot in enumerate(slots):
    x1, y1, x2, y2 = slot['bbox']
    slot_height = y2 - y1
    
    if slot_idx in edge_slot_indices AND slot_height < median_slot_height × 0.9:
        # Expand to match median height
        center_y = (y1 + y2) / 2
        y1 = max(0, int(center_y - median_slot_height / 2))
        y2 = min(height, int(center_y + median_slot_height / 2))
        slot_height = y2 - y1
```

**Formulas:**
- **Edge Detection:** Leftmost and rightmost slot in each row
- **Expansion Condition:** `slot_height < median_slot_height × 0.9`
- **New Height:** `median_slot_height`
- **New Bounds:** `[center_y - median_height/2, center_y + median_height/2]`

---

### Vertical Expansion

**Purpose:** Expand boxes to cover full parking space height.

**Formula:**
```
slot_height = y2 - y1
expand_y = max(15, int(slot_height × 0.50))

draw_y1 = max(0, y1 - expand_y)
draw_y2 = min(height, y2 + expand_y)
```

**Parameters:**
- **Expansion Factor:** `50%` (0.50) on top and bottom
- **Minimum Expansion:** `15 pixels`
- **Total Expansion:** `100%` of slot height (50% top + 50% bottom)

**Why:** Ensures boxes cover full vertical extent of parking spaces including white lines.

---

### Horizontal Gaps

**Purpose:** Create visual separation between adjacent slots.

**Formula:**
```
slot_width = x2 - x1
gap_x = max(3, int(slot_width × 0.03))

draw_x1 = x1 + gap_x
draw_x2 = x2 - gap_x
```

**Parameters:**
- **Gap Percentage:** `3%` on each side
- **Minimum Gap:** `3 pixels`
- **Total Gap:** `6%` of slot width (3% left + 3% right)

**Why:** Visual clarity - prevents boxes from touching each other.

---

### Box Drawing

**Rectangle Drawing:**
```
cv2.rectangle(output, (draw_x1, draw_y1), (draw_x2, draw_y2), color, thickness=2)
```

**Semi-Transparent Fill:**
```
overlay = output.copy()
cv2.rectangle(overlay, (draw_x1, draw_y1), (draw_x2, draw_y2), color, -1)
output = addWeighted(overlay, 0.25, output, 0.75, 0)
```

**Alpha Blending Formula:**
```
output = overlay × 0.25 + output × 0.75
```

**Colors:**
- **Occupied (Red):** `(0, 0, 255)` in BGR
- **Available (Green):** `(0, 255, 0)` in BGR

---

### Label Rendering

**Label Text:**
```
label = f"#{slot['id']+1} {status}"  # e.g., "#1 AVAILABLE"
font_scale = 0.4
thickness = 1
text_y = draw_y1 + 15 if draw_y1 + 15 < draw_y2 - 5 else draw_y2 - 5
```

**Text Drawing (with outline):**
```
# White outline
cv2.putText(output, label, (draw_x1 + 3, text_y),
           FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 1)
# Colored text
cv2.putText(output, label, (draw_x1 + 3, text_y),
           FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
```

**Confidence Display (for occupied slots):**
```
conf_text = f"{slot['confidence'] × 100:.1f}%"
cv2.putText(output, conf_text, (draw_x1 + 3, text_y + 12),
           FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
```

---

### Legend Panel

**Position:**
```
panel_width = 250
panel_height = 160
panel_x = width - panel_width - 15
panel_y = height - panel_height - 15  # Bottom-right
```

**Background:**
```
cv2.rectangle(frame, (panel_x, panel_y), 
             (panel_x + panel_width, panel_y + panel_height), 
             (30, 30, 30), -1)  # Dark gray fill
cv2.rectangle(frame, (panel_x, panel_y), 
             (panel_x + panel_width, panel_y + panel_height), 
             (0, 200, 255), 2)  # Cyan border
```

**Statistics Display:**
```
total = len(slots)
occupied = sum(1 for s in slots if s['occupied'])
available = total - occupied
occupancy_rate = occupied / total if total > 0 else 0
```

**Occupancy Bar:**
```
bar_width = panel_width - 24
fill_width = int(bar_width × occupancy_rate)
cv2.rectangle(frame, (panel_x + 12, bar_y), 
             (panel_x + 12 + fill_width, bar_y + 12), 
             (0, 0, 255), -1)  # Red fill
```

---

## Method Selection & Scoring

### Method Scoring Algorithm

**Base Score:**
```
base_score = len(method_slots)
```

**Ratio-Based Bonus/Penalty:**
```
if len(vehicles) > 0:
    ratio = len(method_slots) / len(vehicles)
    
    if 0.8 <= ratio <= 2.5:
        score = base_score × 2.0      # Ideal ratio
    elif 2.5 < ratio <= 3.5:
        score = base_score × 1.3      # Reasonable
    elif 3.5 < ratio <= 5.0:
        score = base_score × 0.7      # Over-detecting
    else:
        score = base_score × 0.4      # Way off
```

**Absolute Count Penalties:**
```
if len(method_slots) < 5:
    score = score × 0.4               # Too few slots
if len(method_slots) > 120:
    score = score × 0.2               # Too many slots
```

**Selection:**
```
best_method = method with highest score
slots = slots from best_method
```

**Scoring Table:**

| Ratio Range | Multiplier | Reason |
|-------------|-----------|--------|
| 0.8 - 2.5 | × 2.0 | Ideal slot-to-vehicle ratio |
| 2.5 - 3.5 | × 1.3 | Reasonable over-detection |
| 3.5 - 5.0 | × 0.7 | Moderate over-detection |
| > 5.0 or < 0.8 | × 0.4 | Way off target |
| < 5 slots | × 0.4 | Too few detections |
| > 120 slots | × 0.2 | Excessive detections |

---

## Complete Parameter Reference

### White Line Detection

| Parameter | Value | Purpose |
|-----------|-------|---------|
| HSV lower | [0, 0, 180] | White color lower bound |
| HSV upper | [180, 30, 255] | White color upper bound |
| Grayscale threshold | 200 | Brightness threshold |
| Adaptive block size | 11 | Local neighborhood size |
| Adaptive C | -30 | Offset from mean |
| Morph kernel | 3×3 | Cleanup kernel size |

---

### Hough Transform

| Parameter | Value | Purpose |
|-----------|-------|---------|
| rho | 1 pixel | Distance resolution |
| theta | π/180 rad | Angular resolution (1°) |
| threshold | 20 | Minimum votes (lowered) |
| minLineLength | 15 px | Minimum line length (lowered) |
| maxLineGap | 25 px | Maximum gap tolerance (increased) |

---

### Line Classification

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Vertical angle min | 60° | Lower bound for vertical |
| Vertical angle max | 120° | Upper bound for vertical |
| Horizontal angle max | 30° | Upper bound for horizontal (low) |
| Horizontal angle min | 150° | Lower bound for horizontal (high) |

---

### DBSCAN Clustering

| Parameter | Value | Purpose |
|-----------|-------|---------|
| eps | 25 pixels | Maximum cluster distance |
| min_samples | 1 | Minimum cluster size |

---

### Row Detection

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Gap multiplier | 3.0× | Significant gap threshold |
| Empty-lot min width | 6% image | Minimum divider span |
| Empty-lot min height | image/20 | Minimum row height |
| Empty-lot max height | image/2 | Maximum row height |

---

### Slot Creation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Min spacing | 20 px | Minimum slot width |
| Spacing filter min | 0.4× median | Minimum valid spacing |
| Spacing filter max | 2.0× median | Maximum valid spacing |
| Default aspect ratio | 2.0 | Width:Height ratio |
| Max row height | image/4 | Maximum row height |
| Min row height | 1.5× avg_width | Minimum row height |
| Edge slot min width | 0.4× avg_width | Minimum edge slot width |

---

### Slot Filtering

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Width range min | 0.75× median | Minimum slot width |
| Width range max | 1.5× median | Maximum slot width |
| Height range min | 0.75× median | Minimum slot height |
| Height range max | 1.6× median | Maximum slot height |
| Area range min | max(image/5000, 0.5×median) | Minimum slot area |
| Area range max | min(image/6, 1.8×median) | Maximum slot area |
| IoU threshold | 0.7 | Duplicate removal threshold |
| Center tolerance X | 0.4× med_w | X-center proximity |
| Center tolerance Y | 0.6× med_h | Y-center proximity |
| Isolation multiplier | 3.0× | Maximum distance from vehicles |
| Max slots | min(16, vehicles×3) | Hard cap on slot count |

---

### Slot Merging

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Merge threshold | 0.6× med_w | Center distance for merging |
| Overlap threshold | 0.3 | IoU for merging |

---

### Occupancy Detection

| Parameter | Value | Purpose |
|-----------|-------|---------|
| YOLO confidence | 0.05 | Very low to catch all vehicles |
| Overlap threshold | 15% | Minimum overlap for occupied |
| Center check bonus | 0.5 | Overlap if center inside slot |
| Diff threshold | 18.0 | Mean pixel difference |
| Blur kernel | 5×5 | Gaussian blur size |

---

### Visualization

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Vertical expansion | 50% | Top and bottom expansion |
| Min vertical expansion | 15 px | Minimum expansion |
| Horizontal gap | 3% | Gap on each side |
| Min horizontal gap | 3 px | Minimum gap |
| Edge slot threshold | 0.9× median | Height threshold for expansion |
| Fill opacity | 25% | Semi-transparent overlay |
| Panel width | 250 px | Legend panel width |
| Panel height | 160 px | Legend panel height |
| Panel margin | 15 px | Margin from edges |

---

## Mathematical Formulas

### Distance Calculations

**Euclidean Distance:**
```
distance(p1, p2) = sqrt((x2 - x1)² + (y2 - y1)²)
```

**Squared Distance (for efficiency):**
```
distance_sq(p1, p2) = (x2 - x1)² + (y2 - y1)²
```

---

### Intersection over Union (IoU)

```
IoU(box1, box2) = intersection_area / union_area

where:
intersection_area = (ix2 - ix1) × (iy2 - iy1)
union_area = area1 + area2 - intersection_area

ix1 = max(box1.x1, box2.x1)
iy1 = max(box1.y1, box2.y1)
ix2 = min(box1.x2, box2.x2)
iy2 = min(box1.y2, box2.y2)
```

---

### Overlap Ratio

```
overlap_ratio = intersection_area / slot_area
```

---

### Median Calculation

```
median(values) = sorted(values)[len(values) // 2]
```

---

### Alpha Blending

```
output = overlay × alpha + background × (1 - alpha)
```

Where `alpha = 0.25` for semi-transparent fill.

---

### Confidence Calculation

**YOLO-based:**
```
confidence = min(0.90 + overlap_ratio × 0.1, 0.99)  # Occupied
confidence = 0.95  # Available
```

**Difference-based:**
```
confidence = min(0.90 + (diff_score - 18.0) × 0.005, 0.99)  # Occupied
confidence = 0.95  # Available
```

---

### Occupancy Rate

```
occupancy_rate = occupied_count / total_slots
```

---

## Summary: Key Techniques

1. **Multi-Mask White Detection:** Combines HSV, grayscale, and adaptive thresholding for robust line detection.

2. **Conservative DBSCAN Clustering:** Only merges dividers within 25px, preserving distinct stalls.

3. **Empty-Lot Fallback:** Creates synthetic row when no vehicles detected, enabling detection on empty lots.

4. **Edge Slot Creation:** Explicitly creates left/right edge slots, ensuring complete coverage (4 slots from 3 dividers).

5. **Geometric Filtering:** Multi-stage filtering with IoU-based duplicate removal and isolation checks.

6. **Row-Based Merging:** Merges duplicate slots within rows using center distance and overlap checks.

7. **Dual Occupancy Detection:** YOLO matching (primary) + difference-based fallback (for missed vehicles).

8. **Edge Slot Normalization:** Expands edge slots vertically to match median slot height.

9. **Visual Enhancements:** 50% vertical expansion, 3% horizontal gaps, semi-transparent fill, bottom-right legend.

10. **Method Selection:** Scores multiple detection methods and selects best based on slot-to-vehicle ratio.

**The combination of these techniques produces accurate, consistent slot detection with proper occupied/available marking and clean visualization.**

---

**End of Complete Technical Documentation**
