# Parking Slot Detection Formulas

This document describes the formulas currently used by the code for parking slot geometry and occupancy detection. It is intended to replace the simplified formulas shown in the presentation.

## 1. Slot Representation

Each parking slot is represented as a four-point polygon, not just a simple rectangle.

```text
Slot S = [P1, P2, P3, P4]

P1 = (x1, y1)
P2 = (x2, y2)
P3 = (x3, y3)
P4 = (x4, y4)
```

The code also stores a bounding box around the polygon for fast lookup:

```text
bbox(S) = (min_x, min_y, max_x, max_y)
```

where:

```text
min_x = min(x1, x2, x3, x4)
max_x = max(x1, x2, x3, x4)
min_y = min(y1, y2, y3, y4)
max_y = max(y1, y2, y3, y4)
```

Code reference:

```text
src/smart_parking/slots/geometry.py
```

## 2. Vehicle Bounding Box Padding

YOLO detects vehicles as bounding boxes:

```text
B = (x1, y1, x2, y2)
```

Before checking overlap with a parking slot, the code slightly expands the vehicle box to tolerate small detection or alignment errors.

```text
pad_x = (x2 - x1) * pad_ratio
pad_y = (y2 - y1) * pad_ratio
```

The padded vehicle bounding box is:

```text
B' = (
  x1 - pad_x,
  y1 - pad_y,
  x2 + pad_x,
  y2 + pad_y
)
```

In the current backend runtime:

```text
pad_ratio = 0.03
```

Code reference:

```text
src/smart_parking/detection/manual_detector.py
src/smart_parking/api/video_backend.py
```

## 3. Slot Occupancy Overlap Formula

The code calculates how much of a parking slot polygon is covered by the padded vehicle box.

```text
Overlap(S, B') = Area(S intersect B') / Area(S)
```

Where:

```text
S  = parking slot polygon
B' = padded vehicle bounding box
```

This is not standard IoU, because the denominator is only the slot area, not the union area.

Standard IoU would be:

```text
IoU = Area(S intersect B') / Area(S union B')
```

But the code uses:

```text
Slot Overlap Ratio = Area(S intersect B') / Area(S)
```

Code reference:

```text
src/smart_parking/detection/manual_detector.py
```

## 4. Occupancy Decision Rule

The slot is first checked using the overlap ratio:

```text
if Overlap(S, B') > overlap_threshold:
    candidate_status = occupied
```

Current backend value:

```text
overlap_threshold = 0.26
```

If the overlap threshold is not passed, the code uses a fallback rule: if the center point of the detected vehicle is inside the slot polygon, the slot is still treated as an occupied candidate.

Vehicle center:

```text
center(B) = (
  (x1 + x2) / 2,
  (y1 + y2) / 2
)
```

Fallback rule:

```text
if center(B) is inside S:
    candidate_status = occupied
    slot_confidence = 0.50
```

Code reference:

```text
src/smart_parking/detection/manual_detector.py
```

## 5. Slot Confidence Formula

In the current implementation, the displayed slot confidence is not the YOLO model confidence.

The YOLO confidence is used only to filter vehicle detections. After that, the slot confidence is based on geometric overlap:

```text
slot_confidence = Overlap(S, B')
```

or, when the center-point fallback is used:

```text
slot_confidence = 0.50
```

Presentation-friendly wording:

```text
Slot Confidence = Area(slot polygon intersect padded vehicle box) / Area(slot polygon)
```

## 6. Unsure / Occupied / Available Status Rule

After the slot confidence is calculated, the code assigns the final status using the unsure threshold.

Current backend value:

```text
unsure_threshold = 0.55
```

Status rule:

```text
if 0 < slot_confidence < unsure_threshold:
    status = unsure
elif candidate_status == occupied:
    status = occupied
else:
    status = available
```

With current values:

```text
if 0 < slot_confidence < 0.55:
    status = unsure
elif candidate_status == occupied:
    status = occupied
else:
    status = available
```

Important detail:

```text
overlap_threshold = 0.26
unsure_threshold = 0.55
```

This means an overlap above `0.26` creates an occupied candidate, but if the confidence is still below `0.55`, the final displayed status becomes `unsure`.

## 7. YOLO Detection Filter

The backend runs YOLO and accepts detections above a model confidence threshold.

Current backend value:

```text
YOLO confidence threshold = 0.01
```

For vehicle detection, the main accepted COCO classes are:

```text
car        = 2
motorcycle = 3
bus        = 5
truck      = 7
```

If no vehicles are found using those classes, the code has a fallback that accepts detections based on bounding-box size and aspect ratio:

```text
area = box_width * box_height
aspect = box_width / box_height
```

Fallback filters:

```text
image_area / 5000 <= area <= image_area / 6
0.5 <= aspect <= 3.5
```

## 8. Anchor-Based Slot Geometry

The generated slots are based on anchor slots. An anchor slot is a manually labeled quadrilateral with row metadata:

```text
anchor = {
  row,
  slot_index,
  slot_count,
  points
}
```

For the current cropped parking layout, each row has:

```text
slot_count = 12
```

and the seed file uses anchor slots:

```text
slot 1
slot 6
slot 12
```

for each row.

Code reference:

```text
data/slots/parking_cropped_anchor_seeds.json
src/smart_parking/slots/generator.py
```

## 9. Row Direction Formula

For each row, the code estimates the row direction from the center of the first anchor slot to the center of the last anchor slot.

Slot center:

```text
C(S) = (
  average(x1, x2, x3, x4),
  average(y1, y2, y3, y4)
)
```

Row direction vector:

```text
d = normalize(C(last_anchor) - C(first_anchor))
```

where:

```text
normalize(v) = v / ||v||
```

and:

```text
||v|| = sqrt(vx^2 + vy^2)
```

The row direction is used to determine which opposite edges of each anchor are the slot divider boundaries.

## 10. Divider Boundary Lines From Anchors

Each anchor slot produces two divider boundary lines:

```text
D(i - 1)
D(i)
```

where:

```text
i = anchor slot index
```

For example, anchor slot `6` gives boundaries:

```text
D(5)
D(6)
```

The code chooses the pair of opposite edges whose center-to-center direction best matches the row direction.

Edge alignment score:

```text
alignment = abs(dot(normalize(edge_center_2 - edge_center_1), row_direction))
```

The edge pair with the highest alignment is used as the divider pair.

## 11. Projective Interpolation for Missing Boundaries

The code fills missing divider boundaries using a projective interpolation model.

For a row with `N` slots, there are `N + 1` divider boundaries:

```text
D(0), D(1), D(2), ..., D(N)
```

For each boundary index `k`:

```text
u_k = k / N
```

The projective parameter is:

```text
t_k = (a * u_k + b) / (c * u_k + 1)
```

The values `a`, `b`, and `c` are fitted from the known anchor boundary positions using least squares.

If the projective fit fails, the code falls back to piecewise linear interpolation between known anchors.

Code reference:

```text
src/smart_parking/slots/generator.py
```

## 12. Generated Divider Line Formula

After `t_k` is calculated, each missing divider line is interpolated between the first and last row boundaries.

Let:

```text
D(0) = first row boundary
D(N) = last row boundary
```

Each divider line has two endpoints:

```text
D(k).left
D(k).right
```

Generated divider endpoints:

```text
D(k).left =
  D(0).left + t_k * (D(N).left - D(0).left)

D(k).right =
  D(0).right + t_k * (D(N).right - D(0).right)
```

Explicit anchor-derived boundaries are preserved exactly and overwrite the interpolated result.

## 13. Generated Slot Polygon Formula

Slot `k` is created from two neighboring divider lines:

```text
Slot_k = [
  D(k - 1).left,
  D(k - 1).right,
  D(k).right,
  D(k).left
]
```

This is why the slots can follow perspective distortion instead of being fixed-size rectangles.

## 14. Slot Width and Height From Generated Geometry

Because slots are quadrilaterals, width and height are calculated from edge lengths, not from a single rectangle width and height.

Distance between two points:

```text
distance(P, Q) = sqrt((Qx - Px)^2 + (Qy - Py)^2)
```

For generated slot `k`:

```text
slot_width_k =
  (
    distance(D(k).left, D(k - 1).left)
    + distance(D(k).right, D(k - 1).right)
  ) / 2
```

```text
slot_height_k =
  (
    distance(D(k - 1).left, D(k - 1).right)
    + distance(D(k).left, D(k).right)
  ) / 2
```

Presentation-friendly wording:

```text
The slot width is the average distance between neighboring divider boundaries.
The slot height is the average length of the two divider lines that bound the slot.
```

## 15. Recommended Presentation Formulas

Use these formulas in the presentation instead of the current simplified version.

### Parking Slot Polygon

```text
Slot S = [P1, P2, P3, P4]
```

### Vehicle Box Padding

```text
B' = (
  x1 - (x2 - x1) * pad_ratio,
  y1 - (y2 - y1) * pad_ratio,
  x2 + (x2 - x1) * pad_ratio,
  y2 + (y2 - y1) * pad_ratio
)
```

### Slot Overlap Ratio

```text
Overlap Ratio = Area(S intersect B') / Area(S)
```

### Slot Confidence

```text
Slot Confidence = Overlap Ratio
```

### Occupancy Candidate Rule

```text
Occupied Candidate = Overlap Ratio > 0.26
```

### Final Status Rule

```text
if 0 < Slot Confidence < 0.55:
    status = unsure
elif Occupied Candidate:
    status = occupied
else:
    status = available
```

### Anchor-Based Slot Generation

```text
t_k = (a * (k / N) + b) / (c * (k / N) + 1)
```

```text
D(k).left = D(0).left + t_k * (D(N).left - D(0).left)
D(k).right = D(0).right + t_k * (D(N).right - D(0).right)
```

```text
Slot_k = [
  D(k - 1).left,
  D(k - 1).right,
  D(k).right,
  D(k).left
]
```

