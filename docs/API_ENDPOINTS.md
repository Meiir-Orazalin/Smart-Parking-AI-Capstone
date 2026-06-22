# SmartPark API Endpoints (Current)

Base URL: `http://localhost:5001` (or your tunnel URL)

Tunnel URL behavior:
- Cloudflare quick tunnels change on every run.
- Without a Cloudflare zone/domain, the public URL is not fixed.

All responses are wrapped.

## GET /status

```json
{
  "status": {
    "rtsp_url": "https://.../mjpeg",
    "has_frame": true,
    "frame_count": 1024,
    "fps_estimate": 23.4,
    "last_frame_age_seconds": 0.4
  }
}
```

## GET /lots

```json
{
  "lots": [
    {
      "id": "main-lot",
      "name": "Main Lot",
      "occupied": 15,
      "capacity": 69,
      "distanceMi": 0.3,
      "status": "available",
      "lastUpdated": "2026-02-15T20:30:56Z"
    }
  ]
}
```

## GET /dashboard

```json
{
  "dashboard": {
    "totalLots": 1,
    "occupancyPct": 74,
    "activeAlerts": 0,
    "systemFps": 23.5
  }
}
```

## GET /alerts

```json
{
  "alerts": [
    {
      "id": "alert-01",
      "title": "Camera Online",
      "severity": "low",
      "location": "Main Lot",
      "timeAgo": "Just now"
    }
  ]
}
```

## GET /cameras

```json
{
  "cameras": [
    {
      "id": "cam-01",
      "name": "Main Lot Camera",
      "fps": 24,
      "uptime": "99.2%",
      "lastDetection": "Just now",
      "status": "online",
      "streamUrl": "https://.../mjpeg"
    }
  ]
}
```

## POST /cameras

Request:
```json
{
  "name": "Entrance Cam 01",
  "location": "North Lot Gate",
  "streamUrl": "rtsp://operator:pass@192.168.1.80:554/stream1"
}
```

Response:
```json
{
  "camera": {
    "id": "cam-123",
    "name": "Entrance Cam 01 - North Lot Gate",
    "fps": 0,
    "uptime": "100.0%",
    "lastDetection": "Just now",
    "status": "online",
    "streamUrl": "rtsp://operator:pass@192.168.1.80:554/stream1"
  }
}
```

## GET /profile

```json
{
  "profile": {
    "name": "SmartPark Demo",
    "email": "demo@smartpark.local",
    "notificationsEnabled": true
  }
}
```
