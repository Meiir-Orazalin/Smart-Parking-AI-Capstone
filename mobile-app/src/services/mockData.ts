import {
  AlertItem,
  CameraItem,
  DashboardStats,
  ParkingLot,
  UserProfile
} from "../types/models";

function isoMinutesAgo(minutes: number): string {
  return new Date(Date.now() - minutes * 60 * 1000).toISOString();
}

export const lotsMock: ParkingLot[] = [
  {
    id: "north-lot",
    name: "North Lot",
    occupied: 15,
    capacity: 50,
    distanceMi: 0.3,
    status: "available",
    lastUpdated: isoMinutesAgo(1)
  },
  {
    id: "central-garage",
    name: "Central Garage",
    occupied: 88,
    capacity: 100,
    distanceMi: 0.8,
    status: "almost_full",
    lastUpdated: isoMinutesAgo(2)
  },
  {
    id: "west-plaza",
    name: "West Plaza",
    occupied: 120,
    capacity: 120,
    distanceMi: 1.1,
    status: "full",
    lastUpdated: isoMinutesAgo(5)
  },
  {
    id: "south-deck",
    name: "South Deck",
    occupied: 42,
    capacity: 80,
    distanceMi: 0.9,
    status: "available",
    lastUpdated: isoMinutesAgo(3)
  },
  {
    id: "visitor-garage-l2",
    name: "Visitor Garage - L2",
    occupied: 72,
    capacity: 90,
    distanceMi: 0.6,
    status: "almost_full",
    lastUpdated: isoMinutesAgo(2)
  },
  {
    id: "west-wing",
    name: "West Wing",
    occupied: 64,
    capacity: 64,
    distanceMi: 1.4,
    status: "full",
    lastUpdated: isoMinutesAgo(7)
  }
];

export const dashboardMock: DashboardStats = {
  totalLots: 4,
  occupancyPct: 78,
  activeAlerts: 3,
  systemFps: 23.5
};

export const alertsMock: AlertItem[] = [
  {
    id: "alert-01",
    title: "Camera Offline",
    severity: "high",
    location: "West Plaza",
    timeAgo: "3 min ago"
  },
  {
    id: "alert-02",
    title: "Loitering Detected",
    severity: "medium",
    location: "North Lot",
    timeAgo: "8 min ago"
  },
  {
    id: "alert-03",
    title: "Unauthorized Vehicle",
    severity: "high",
    location: "Visitor Garage - L2",
    timeAgo: "12 min ago"
  }
];

export const camerasMock: CameraItem[] = [
  {
    id: "cam-01",
    name: "North Gate Camera",
    fps: 24,
    uptime: "99.2%",
    lastDetection: "2 min ago",
    status: "online"
  },
  {
    id: "cam-02",
    name: "Central Ramp Camera",
    fps: 18,
    uptime: "92.8%",
    lastDetection: "1 min ago",
    status: "online"
  },
  {
    id: "cam-03",
    name: "West Plaza Camera",
    fps: 0,
    uptime: "63.1%",
    lastDetection: "42 min ago",
    status: "offline"
  },
  {
    id: "cam-04",
    name: "South Entry Camera",
    fps: 30,
    uptime: "98.5%",
    lastDetection: "30 sec ago",
    status: "online"
  }
];

export const profileMock: UserProfile = {
  name: "Meiir Orazalin",
  email: "meiirorazalin@gmail.com",
  notificationsEnabled: true
};
