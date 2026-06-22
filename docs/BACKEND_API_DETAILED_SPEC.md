# SmartPark Backend API Detailed Specification

This document defines the exact API contract expected by the current mobile app implementation.

## 1) Base Configuration

- Base URL is taken from `EXPO_PUBLIC_SMARTPARK_API_URL`.
- Expected default during local development: `http://localhost:5001`.
- App modes:
  - `mock`: backend not used
  - `hybrid`: backend first, fallback to mock
  - `api`: backend only, no fallback

## 2) Required Endpoints

1. `GET /status`
2. `GET /lots`
3. `GET /dashboard`
4. `GET /alerts`
5. `GET /cameras`
6. `POST /cameras`
7. `GET /profile`

## 3) Wrapper Compatibility Rules

The app accepts both direct and wrapped responses.

1. `GET /lots`:
   - direct: `[...]`
   - wrapped: `{ "lots": [...] }`
2. `GET /alerts`:
   - direct: `[...]`
   - wrapped: `{ "alerts": [...] }`
3. `GET /cameras`:
   - direct: `[...]`
   - wrapped: `{ "cameras": [...] }`
4. `GET /dashboard`:
   - direct: `{ ... }`
   - wrapped: `{ "dashboard": { ... } }`
5. `GET /profile`:
   - direct: `{ ... }`
   - wrapped: `{ "profile": { ... } }`
6. `GET /status`:
   - direct: `{ ... }`
   - wrapped: `{ "status": { ... } }`
7. `POST /cameras`:
   - direct: `{ ...camera }`
   - wrapped: `{ "camera": { ...camera } }`

## 4) Data Model (Exact Types)

```ts
type LotStatus = "available" | "almost_full" | "full";
type AlertSeverity = "low" | "medium" | "high";
type CameraStatus = "online" | "offline";

interface ParkingLotApiItem {
  id: string;
  name: string;
  occupied: number;
  capacity: number;
  distanceMi: number;
  status: LotStatus;
  lastUpdated: string; // ISO datetime
}

interface DashboardStatsApiResponse {
  totalLots: number;
  occupancyPct: number;
  activeAlerts: number;
  systemFps: number;
}

interface AlertApiItem {
  id: string;
  title: string;
  severity: AlertSeverity;
  location: string;
  timeAgo: string; // human readable, e.g. "3 min ago"
}

interface CameraApiItem {
  id: string;
  name: string;
  fps: number;
  uptime: string; // e.g. "99.2%"
  lastDetection: string; // e.g. "2 min ago"
  status: CameraStatus;
  streamUrl?: string;
}

interface CreateCameraApiRequest {
  name: string;
  location: string;
  streamUrl: string;
}

interface UserProfileApiResponse {
  name: string;
  email: string;
  notificationsEnabled: boolean;
}

interface StatusApiResponse {
  rtsp_url: string;
  has_frame: boolean;
  frame_count: number;
  fps_estimate: number;
  last_frame_age_seconds: number | null;
}
```

## 5) Endpoint Details

### 5.1 `GET /status`

Purpose:
- Surface RTSP pipeline health and frame availability.

Required fields:
- `rtsp_url` (string)
- `has_frame` (boolean)
- `frame_count` (number)
- `fps_estimate` (number)
- `last_frame_age_seconds` (number or null)

Example:

```json
{
  "status": {
    "rtsp_url": "rtsp://operator:pass@192.168.1.50:554/stream1",
    "has_frame": true,
    "frame_count": 1024,
    "fps_estimate": 23.4,
    "last_frame_age_seconds": 0.4
  }
}
```

### 5.2 `GET /lots`

Purpose:
- Driver + ops lot occupancy list.

Required per item:
- `id`, `name` (string)
- `occupied`, `capacity`, `distanceMi` (number)
- `status` (`available|almost_full|full`)
- `lastUpdated` (ISO datetime string)

Example:

```json
{
  "lots": [
    {
      "id": "north-lot",
      "name": "North Lot",
      "occupied": 15,
      "capacity": 50,
      "distanceMi": 0.3,
      "status": "available",
      "lastUpdated": "2026-02-14T08:14:00.000Z"
    }
  ]
}
```

### 5.3 `GET /dashboard`

Purpose:
- Operations summary cards.

Required fields:
- `totalLots` (number)
- `occupancyPct` (number)
- `activeAlerts` (number)
- `systemFps` (number)

Example:

```json
{
  "dashboard": {
    "totalLots": 6,
    "occupancyPct": 74,
    "activeAlerts": 3,
    "systemFps": 23.5
  }
}
```

### 5.4 `GET /alerts`

Purpose:
- Operations recent alerts list.

Required per item:
- `id`, `title`, `location`, `timeAgo` (string)
- `severity` (`low|medium|high`)

Example:

```json
{
  "alerts": [
    {
      "id": "alert-01",
      "title": "Camera Offline",
      "severity": "high",
      "location": "West Plaza",
      "timeAgo": "3 min ago"
    }
  ]
}
```

### 5.5 `GET /cameras`

Purpose:
- Camera management list + health state.

Required per item:
- `id`, `name`, `uptime`, `lastDetection` (string)
- `fps` (number)
- `status` (`online|offline`)
- optional `streamUrl` (string)

Example:

```json
{
  "cameras": [
    {
      "id": "cam-01",
      "name": "North Gate Camera",
      "fps": 24,
      "uptime": "99.2%",
      "lastDetection": "2 min ago",
      "status": "online",
      "streamUrl": "rtsp://operator:pass@192.168.1.80:554/stream1"
    }
  ]
}
```

### 5.6 `POST /cameras`

Purpose:
- Add camera from operations UI.

Request body:
- `name` (string, required)
- `location` (string, required)
- `streamUrl` (string, required, should be RTSP URL)

Request example:

```json
{
  "name": "Entrance Cam 01",
  "location": "North Lot Gate",
  "streamUrl": "rtsp://operator:pass@192.168.1.80:554/stream1"
}
```

Response:
- return full `CameraApiItem` object.
- `id` must be stable/unique.

Response example:

```json
{
  "camera": {
    "id": "cam-123",
    "name": "Entrance Cam 01 - North Lot Gate",
    "fps": 24,
    "uptime": "100.0%",
    "lastDetection": "Just now",
    "status": "online",
    "streamUrl": "rtsp://operator:pass@192.168.1.80:554/stream1"
  }
}
```

### 5.7 `GET /profile`

Purpose:
- Driver profile screen.

Required fields:
- `name` (string)
- `email` (string)
- `notificationsEnabled` (boolean)

Example:

```json
{
  "profile": {
    "name": "Meiir Orazalin",
    "email": "meiirorazalin@gmail.com",
    "notificationsEnabled": true
  }
}
```

## 6) Validation Rules (Server Side)

1. Enum values must be exact string matches:
   - lot status: `available|almost_full|full`
   - alert severity: `low|medium|high`
   - camera status: `online|offline`
2. Numeric fields must be JSON numbers, not strings.
3. `lastUpdated` should be ISO datetime.
4. `POST /cameras` should reject invalid `streamUrl` not starting with `rtsp://`.
5. `POST /cameras` should reject duplicates by camera name and/or stream URL when possible.

## 7) Error Handling

Recommended JSON error format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "streamUrl must start with rtsp://"
  }
}
```

Recommended status codes:
- `400` bad request / validation error
- `404` resource not found
- `409` conflict (duplicate camera)
- `500` server error

## 8) Important Compatibility Notes

1. In `api` mode, bad payload shape causes immediate UI failure (no mock fallback).
2. In `hybrid` mode, invalid backend response triggers fallback to local mock data.
3. Avoid returning empty lists unless that state is intentionally handled by your backend strategy.
4. Keep field names and letter casing exactly as above (`distanceMi`, `lastUpdated`, `rtsp_url`, etc.).

## 9) Optional Future Endpoints (Not required yet)

1. `PATCH /alerts/:id/workflow` for shared alert state sync.
2. `PATCH /profile` for notifications preference persistence.
3. Auth endpoints if multi-user sessions are introduced.
