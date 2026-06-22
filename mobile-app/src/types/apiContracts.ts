import {
  AlertSeverity,
  CameraStatus,
  LotStatus
} from "./models";

export interface StatusApiResponse {
  rtsp_url: string;
  has_frame: boolean;
  frame_count: number;
  fps_estimate: number;
  last_frame_age_seconds: number | null;
}

export interface ParkingLotApiItem {
  id: string;
  name: string;
  occupied: number;
  capacity: number;
  available?: number;
  unsure?: number;
  distanceMi: number;
  status: LotStatus;
  lastUpdated: string;
}

export interface DashboardStatsApiResponse {
  totalLots: number;
  occupancyPct: number;
  activeAlerts: number;
  systemFps: number;
}

export interface AlertApiItem {
  id: string;
  title: string;
  severity: AlertSeverity;
  location: string;
  timeAgo: string;
}

export interface CameraApiItem {
  id: string;
  name: string;
  fps: number;
  uptime: string;
  lastDetection: string;
  status: CameraStatus;
  streamUrl?: string;
}

export interface CreateCameraApiRequest {
  name: string;
  location: string;
  streamUrl: string;
}

export interface UserProfileApiResponse {
  name: string;
  email: string;
  notificationsEnabled: boolean;
}

export const apiContractExamples = {
  status: {
    rtsp_url: "rtsp://operator:pass@192.168.1.50:554/stream1",
    has_frame: true,
    frame_count: 1024,
    fps_estimate: 23.4,
    last_frame_age_seconds: 0.4
  } satisfies StatusApiResponse,
  lots: [
    {
      id: "north-lot",
      name: "North Lot",
      occupied: 15,
      capacity: 50,
      distanceMi: 0.3,
      status: "available",
      lastUpdated: "2026-02-14T08:14:00.000Z"
    }
  ] satisfies ParkingLotApiItem[],
  dashboard: {
    totalLots: 6,
    occupancyPct: 74,
    activeAlerts: 3,
    systemFps: 23.5
  } satisfies DashboardStatsApiResponse,
  alerts: [
    {
      id: "alert-01",
      title: "Camera Offline",
      severity: "high",
      location: "West Plaza",
      timeAgo: "3 min ago"
    }
  ] satisfies AlertApiItem[],
  cameras: [
    {
      id: "cam-01",
      name: "North Gate Camera",
      fps: 24,
      uptime: "99.2%",
      lastDetection: "2 min ago",
      status: "online"
    }
  ] satisfies CameraApiItem[],
  createCameraRequest: {
    name: "Entrance Cam 01",
    location: "North Lot Gate",
    streamUrl: "rtsp://operator:pass@192.168.1.80:554/stream1"
  } satisfies CreateCameraApiRequest,
  profile: {
    name: "Meiir Orazalin",
    email: "meiirorazalin@gmail.com",
    notificationsEnabled: true
  } satisfies UserProfileApiResponse
} as const;
