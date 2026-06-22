export type LotStatus = "available" | "almost_full" | "full";
export type AlertSeverity = "low" | "medium" | "high";
export type CameraStatus = "online" | "offline";
export type MockScenario = "normal" | "peak" | "event" | "incident";
export type DataMode = "mock" | "hybrid" | "api";
export type AlertWorkflowStatus = "new" | "acknowledged" | "assigned" | "resolved";

export interface ParkingLot {
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

export interface DashboardStats {
  totalLots: number;
  occupancyPct: number;
  activeAlerts: number;
  systemFps: number;
}

export interface AlertItem {
  id: string;
  title: string;
  severity: AlertSeverity;
  location: string;
  timeAgo: string;
}

export interface CameraItem {
  id: string;
  name: string;
  fps: number;
  uptime: string;
  lastDetection: string;
  status: CameraStatus;
  streamUrl?: string;
}

export interface NewCameraInput {
  name: string;
  location: string;
  streamUrl: string;
}

export interface UserProfile {
  name: string;
  email: string;
  notificationsEnabled: boolean;
}

export interface BackendStatus {
  rtsp_url: string;
  has_frame: boolean;
  frame_count: number;
  fps_estimate: number;
  last_frame_age_seconds: number | null;
}
