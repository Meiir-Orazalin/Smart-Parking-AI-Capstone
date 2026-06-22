import {
  AlertItem,
  BackendStatus,
  CameraItem,
  DashboardStats,
  DataMode,
  MockScenario,
  NewCameraInput,
  ParkingLot,
  UserProfile
} from "../types/models";
import {
  AlertApiItem,
  CameraApiItem,
  DashboardStatsApiResponse,
  ParkingLotApiItem,
  StatusApiResponse,
  UserProfileApiResponse,
  apiContractExamples
} from "../types/apiContracts";
import {
  alertsMock,
  camerasMock,
  dashboardMock,
  lotsMock,
  profileMock
} from "./mockData";
import { readJson, removeJson, writeJson } from "./storage";

const DEFAULT_API_BASE_URL = "http://localhost:5001";
const DEFAULT_DATA_MODE: DataMode = "hybrid";
const API_BASE_URL = (process.env.EXPO_PUBLIC_SMARTPARK_API_URL ?? DEFAULT_API_BASE_URL).replace(/\/$/, "");
const STATUS_ENDPOINT = `${API_BASE_URL}/status`;
const LOTS_ENDPOINT = `${API_BASE_URL}/lots`;
const DASHBOARD_ENDPOINT = `${API_BASE_URL}/dashboard`;
const ALERTS_ENDPOINT = `${API_BASE_URL}/alerts`;
const CAMERAS_ENDPOINT = `${API_BASE_URL}/cameras`;
const PROFILE_ENDPOINT = `${API_BASE_URL}/profile`;
const CAMERA_STORE_KEY = "smartpark:cameras:v1";
const MOCK_DELAY_MS = 600;
const REMOTE_TIMEOUT_MS = 2500;
const DYNAMIC_BUCKET_MS = 30_000;
const MAX_SWING_RATIO = 0.14;
const NGROK_SKIP_BROWSER_WARNING_HEADER = "ngrok-skip-browser-warning";
const NGROK_SKIP_BROWSER_WARNING_VALUE = "true";
const SCENARIO_LIST: MockScenario[] = ["normal", "peak", "event", "incident"];
const MAIN_LOT_DISPLAY_NAME = "RIT Dubai Dormitory";

const scenarioConfig: Record<MockScenario, { occupancyShift: number; swingMultiplier: number; alertBonus: number }> = {
  normal: { occupancyShift: 0, swingMultiplier: 1, alertBonus: 0 },
  peak: { occupancyShift: 0.08, swingMultiplier: 1.2, alertBonus: 2 },
  event: { occupancyShift: 0.05, swingMultiplier: 1.1, alertBonus: 1 },
  incident: { occupancyShift: 0.1, swingMultiplier: 0.75, alertBonus: 3 }
};

let activeScenario: MockScenario = "normal";
let cameraStore: CameraItem[] = camerasMock.map((camera) => ({ ...camera }));
let cameraStoreHydrationPromise: Promise<void> | null = null;
let latestRemoteLots: ParkingLot[] = [];

function normalizeDataMode(value: string | undefined): DataMode {
  const raw = (value ?? DEFAULT_DATA_MODE).toLowerCase();
  if (raw === "api" || raw === "hybrid" || raw === "mock") {
    return raw;
  }
  return DEFAULT_DATA_MODE;
}

const DATA_MODE = normalizeDataMode(process.env.EXPO_PUBLIC_SMARTPARK_DATA_MODE);

export type RuntimePreflightIssue = {
  code: "api_url_missing" | "api_url_invalid" | "api_url_localhost_on_phone" | "contract_mismatch";
  severity: "warning" | "error";
  message: string;
};

function shouldUseRemoteData(): boolean {
  return DATA_MODE !== "mock";
}

function allowMockFallback(): boolean {
  return DATA_MODE !== "api";
}

function isValidHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function cloneLots(data: ParkingLot[]): ParkingLot[] {
  return data.map((lot) => ({ ...lot }));
}

function cloneAlerts(data: AlertItem[]): AlertItem[] {
  return data.map((alert) => ({ ...alert }));
}

function cloneCameras(data: CameraItem[]): CameraItem[] {
  return data.map((camera) => ({ ...camera }));
}

function delay(ms = MOCK_DELAY_MS): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function deriveStatus(occupied: number, capacity: number): ParkingLot["status"] {
  if (capacity <= 0 || occupied >= capacity) return "full";
  const ratio = occupied / capacity;
  if (ratio >= 0.85) return "almost_full";
  return "available";
}

function getCurrentBucket(now = Date.now()): number {
  return Math.floor(now / DYNAMIC_BUCKET_MS);
}

function buildDynamicLots(bucket = getCurrentBucket()): ParkingLot[] {
  const config = scenarioConfig[activeScenario];

  return cloneLots(lotsMock).map((lot, index) => {
    const seed = hashString(lot.id) + index * 17;
    const waveA = Math.sin((bucket + seed) * 0.35);
    const waveB = Math.cos((bucket + seed) * 0.19);
    const combinedWave = waveA * 0.65 + waveB * 0.35;
    const swing = Math.max(2, Math.round(lot.capacity * MAX_SWING_RATIO * config.swingMultiplier));
    const shiftedBase = lot.occupied + Math.round(lot.capacity * config.occupancyShift);
    const occupied = clamp(shiftedBase + Math.round(combinedWave * swing), 0, lot.capacity);
    const freshnessOffset = Math.abs((seed + bucket) % 4);
    const lastUpdated = new Date((bucket - freshnessOffset) * DYNAMIC_BUCKET_MS).toISOString();

    return {
      ...lot,
      occupied,
      status: deriveStatus(occupied, lot.capacity),
      lastUpdated
    };
  });
}

function isMainLot(lot: ParkingLot): boolean {
  const normalizedName = lot.name.trim().toLowerCase();
  return lot.id === "main-lot" || normalizedName === "main lot" || normalizedName === MAIN_LOT_DISPLAY_NAME.toLowerCase();
}

function mergeRemoteLotsWithFallback(remoteLots: ParkingLot[], fallbackLots = buildDynamicLots()): ParkingLot[] {
  const seen = new Set<string>();
  const merged: ParkingLot[] = [];

  [...remoteLots, ...fallbackLots].forEach((lot) => {
    const key = `${lot.id.trim().toLowerCase()}|${lot.name.trim().toLowerCase()}`;
    if (seen.has(key)) {
      return;
    }

    seen.add(key);
    merged.push({ ...lot });
  });

  return merged.sort((a, b) => {
    if (isMainLot(a)) return -1;
    if (isMainLot(b)) return 1;
    return 0;
  });
}

function withNgrokBypassHeader(headers?: HeadersInit): Headers {
  const normalized = new Headers(headers);
  if (!normalized.has(NGROK_SKIP_BROWSER_WARNING_HEADER)) {
    normalized.set(NGROK_SKIP_BROWSER_WARNING_HEADER, NGROK_SKIP_BROWSER_WARNING_VALUE);
  }
  return normalized;
}

async function fetchJson<T>(
  url: string,
  init?: RequestInit,
  timeoutMs = REMOTE_TIMEOUT_MS
): Promise<T | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...init,
      headers: withNgrokBypassHeader(init?.headers),
      signal: controller.signal
    });

    if (!response.ok) {
      return null;
    }

    return (await response.json()) as T;
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

function parseApiArray(payload: unknown, collectionKey: string): unknown[] | null {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (payload && typeof payload === "object") {
    const collection = (payload as Record<string, unknown>)[collectionKey];
    if (Array.isArray(collection)) {
      return collection;
    }
  }

  return null;
}

function parseApiObject(payload: unknown, objectKey: string): Record<string, unknown> | null {
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    if (objectKey in payload) {
      const nested = (payload as Record<string, unknown>)[objectKey];
      if (nested && typeof nested === "object" && !Array.isArray(nested)) {
        return nested as Record<string, unknown>;
      }
    }

    return payload as Record<string, unknown>;
  }

  return null;
}

function normalizeLot(payload: unknown): ParkingLot | null {
  const data = payload as Partial<ParkingLotApiItem>;

  if (
    !data ||
    typeof data !== "object" ||
    typeof data.id !== "string" ||
    typeof data.name !== "string" ||
    typeof data.occupied !== "number" ||
    typeof data.capacity !== "number" ||
    typeof data.distanceMi !== "number" ||
    typeof data.lastUpdated !== "string"
  ) {
    return null;
  }

  const status =
    data.status === "available" || data.status === "almost_full" || data.status === "full"
      ? data.status
      : deriveStatus(data.occupied, data.capacity);

  return {
    id: data.id,
    name: isMainLot({ id: data.id, name: data.name } as ParkingLot) ? MAIN_LOT_DISPLAY_NAME : data.name,
    occupied: data.occupied,
    capacity: data.capacity,
    available: typeof data.available === "number" ? data.available : undefined,
    unsure: typeof data.unsure === "number" ? data.unsure : undefined,
    distanceMi: data.distanceMi,
    status,
    lastUpdated: data.lastUpdated
  };
}

function normalizeDashboard(payload: unknown): DashboardStats | null {
  const data = payload as Partial<DashboardStatsApiResponse>;

  if (
    !data ||
    typeof data !== "object" ||
    typeof data.totalLots !== "number" ||
    typeof data.occupancyPct !== "number" ||
    typeof data.activeAlerts !== "number" ||
    typeof data.systemFps !== "number"
  ) {
    return null;
  }

  return {
    totalLots: data.totalLots,
    occupancyPct: data.occupancyPct,
    activeAlerts: data.activeAlerts,
    systemFps: data.systemFps
  };
}

function normalizeAlert(payload: unknown): AlertItem | null {
  const data = payload as Partial<AlertApiItem>;

  if (
    !data ||
    typeof data !== "object" ||
    typeof data.id !== "string" ||
    typeof data.title !== "string" ||
    typeof data.location !== "string" ||
    typeof data.timeAgo !== "string" ||
    (data.severity !== "low" && data.severity !== "medium" && data.severity !== "high")
  ) {
    return null;
  }

  return {
    id: data.id,
    title: data.title,
    severity: data.severity,
    location: data.location,
    timeAgo: data.timeAgo
  };
}

function normalizeCamera(payload: unknown): CameraItem | null {
  const data = payload as Partial<CameraApiItem>;

  if (
    !data ||
    typeof data !== "object" ||
    typeof data.id !== "string" ||
    typeof data.name !== "string" ||
    typeof data.fps !== "number" ||
    typeof data.uptime !== "string" ||
    typeof data.lastDetection !== "string" ||
    (data.status !== "online" && data.status !== "offline")
  ) {
    return null;
  }

  return {
    id: data.id,
    name: data.name,
    fps: data.fps,
    uptime: data.uptime,
    lastDetection: data.lastDetection,
    status: data.status,
    streamUrl: typeof (payload as { streamUrl?: unknown }).streamUrl === "string"
      ? (payload as { streamUrl: string }).streamUrl
      : undefined
  };
}

function normalizeProfile(payload: unknown): UserProfile | null {
  const data = payload as Partial<UserProfileApiResponse>;

  if (
    !data ||
    typeof data !== "object" ||
    typeof data.name !== "string" ||
    typeof data.email !== "string" ||
    typeof data.notificationsEnabled !== "boolean"
  ) {
    return null;
  }

  return {
    name: data.name,
    email: data.email,
    notificationsEnabled: data.notificationsEnabled
  };
}

function normalizeStatus(payload: unknown): BackendStatus | null {
  const data = payload as Partial<StatusApiResponse & { video_path?: string }>;

  if (
    !data ||
    typeof data !== "object" ||
    typeof data.has_frame !== "boolean" ||
    typeof data.frame_count !== "number" ||
    typeof data.fps_estimate !== "number" ||
    !(typeof data.last_frame_age_seconds === "number" || data.last_frame_age_seconds === null)
  ) {
    return null;
  }

  const rtsp_url =
    typeof data.rtsp_url === "string" ? data.rtsp_url : typeof data.video_path === "string" ? data.video_path : "";

  return {
    rtsp_url,
    has_frame: data.has_frame,
    frame_count: data.frame_count,
    fps_estimate: data.fps_estimate,
    last_frame_age_seconds: data.last_frame_age_seconds
  };
}

async function ensureCameraStoreReady(): Promise<void> {
  if (!cameraStoreHydrationPromise) {
    cameraStoreHydrationPromise = (async () => {
      const persisted = await readJson<unknown>(CAMERA_STORE_KEY, null);

      if (Array.isArray(persisted)) {
        const normalized = persisted
          .map((item) => normalizeCamera(item))
          .filter((item): item is CameraItem => item !== null);

        if (normalized.length > 0) {
          cameraStore = normalized;
        }
      }
    })();
  }

  await cameraStoreHydrationPromise;
}

async function persistCameraStore(): Promise<void> {
  await writeJson(CAMERA_STORE_KEY, cameraStore);
}

async function fetchRemoteStatus(): Promise<BackendStatus | null> {
  const payload = await fetchJson<unknown>(STATUS_ENDPOINT);
  if (!payload) return null;

  const statusData = parseApiObject(payload, "status");
  if (!statusData) return null;

  return normalizeStatus(statusData);
}

async function fetchRemoteLots(): Promise<ParkingLot[] | null> {
  const payload = await fetchJson<unknown>(LOTS_ENDPOINT);
  const rows = parseApiArray(payload, "lots");
  if (!rows) return null;

  const normalized = rows
    .map((item) => normalizeLot(item))
    .filter((item): item is ParkingLot => item !== null);

  if (normalized.length === 0) {
    return null;
  }

  return normalized;
}

async function fetchRemoteDashboard(): Promise<DashboardStats | null> {
  const payload = await fetchJson<unknown>(DASHBOARD_ENDPOINT);
  if (!payload) return null;

  const dashboardData = parseApiObject(payload, "dashboard");
  if (!dashboardData) return null;

  return normalizeDashboard(dashboardData);
}

async function fetchRemoteAlerts(): Promise<AlertItem[] | null> {
  const payload = await fetchJson<unknown>(ALERTS_ENDPOINT);
  const rows = parseApiArray(payload, "alerts");
  if (!rows) return null;

  const normalized = rows
    .map((item) => normalizeAlert(item))
    .filter((item): item is AlertItem => item !== null);

  if (normalized.length === 0) {
    return null;
  }

  return normalized;
}

async function fetchRemoteCameras(): Promise<CameraItem[] | null> {
  const payload = await fetchJson<unknown>(CAMERAS_ENDPOINT);
  const rows = parseApiArray(payload, "cameras");
  if (!rows) return null;

  const normalized = rows
    .map((item) => normalizeCamera(item))
    .filter((item): item is CameraItem => item !== null);

  if (normalized.length === 0) {
    return null;
  }

  return normalized;
}

async function createRemoteCamera(input: NewCameraInput): Promise<CameraItem | null> {
  const payload = await fetchJson<unknown>(
    CAMERAS_ENDPOINT,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(input)
    }
  );

  if (!payload) return null;
  const responseData = parseApiObject(payload, "camera");
  if (!responseData) return null;

  const normalized = normalizeCamera(responseData);
  if (!normalized) return null;

  if (!normalized.streamUrl) {
    return {
      ...normalized,
      streamUrl: input.streamUrl
    };
  }

  return normalized;
}

async function fetchRemoteProfile(): Promise<UserProfile | null> {
  const payload = await fetchJson<unknown>(PROFILE_ENDPOINT);
  if (!payload) return null;

  const profileData = parseApiObject(payload, "profile");
  if (!profileData) return null;

  return normalizeProfile(profileData);
}

function throwWhenApiStrict(message: string): void {
  if (!allowMockFallback()) {
    throw new Error(message);
  }
}

export function getDataMode(): DataMode {
  return DATA_MODE;
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export function isLocalOnlyDataMode(): boolean {
  return DATA_MODE === "mock" || DATA_MODE === "hybrid";
}

export function runApiContractSanityCheck(): { ok: boolean; failures: string[] } {
  const failures: string[] = [];

  if (!normalizeStatus(apiContractExamples.status)) {
    failures.push("status");
  }

  if (!normalizeDashboard(apiContractExamples.dashboard)) {
    failures.push("dashboard");
  }

  if (!normalizeProfile(apiContractExamples.profile)) {
    failures.push("profile");
  }

  if (normalizeLot(apiContractExamples.lots[0]) === null) {
    failures.push("lots[0]");
  }

  if (normalizeAlert(apiContractExamples.alerts[0]) === null) {
    failures.push("alerts[0]");
  }

  if (normalizeCamera(apiContractExamples.cameras[0]) === null) {
    failures.push("cameras[0]");
  }

  return {
    ok: failures.length === 0,
    failures
  };
}

export function getRuntimePreflightIssues(): RuntimePreflightIssue[] {
  const issues: RuntimePreflightIssue[] = [];
  const apiUrlRaw = process.env.EXPO_PUBLIC_SMARTPARK_API_URL;
  const apiUrl = (apiUrlRaw ?? "").trim();

  if (DATA_MODE !== "mock") {
    if (!apiUrl) {
      issues.push({
        code: "api_url_missing",
        severity: "error",
        message:
          "API mode requires EXPO_PUBLIC_SMARTPARK_API_URL. Falling back is only allowed in hybrid mode."
      });
    } else if (!isValidHttpUrl(apiUrl)) {
      issues.push({
        code: "api_url_invalid",
        severity: "error",
        message: "EXPO_PUBLIC_SMARTPARK_API_URL must be a valid http/https URL."
      });
    } else if (apiUrl.includes("localhost") || apiUrl.includes("127.0.0.1")) {
      issues.push({
        code: "api_url_localhost_on_phone",
        severity: "warning",
        message: "localhost/127.0.0.1 will not work from physical phones. Use LAN IP or public host."
      });
    }
  }

  const contractCheck = runApiContractSanityCheck();
  if (!contractCheck.ok) {
    issues.push({
      code: "contract_mismatch",
      severity: "error",
      message: `API contract sanity check failed for: ${contractCheck.failures.join(", ")}`
    });
  }

  return issues;
}

export function listMockScenarios(): MockScenario[] {
  return [...SCENARIO_LIST];
}

export function getMockScenario(): MockScenario {
  return activeScenario;
}

export function setMockScenario(scenario: MockScenario): void {
  activeScenario = scenario;
}

export async function getParkingLots(): Promise<ParkingLot[]> {
  if (shouldUseRemoteData()) {
    const remoteLots = await fetchRemoteLots();
    if (remoteLots) {
      latestRemoteLots = cloneLots(remoteLots);
      return mergeRemoteLotsWithFallback(remoteLots);
    }

    throwWhenApiStrict("Unable to load parking lots from API.");

    if (latestRemoteLots.length > 0) {
      return mergeRemoteLotsWithFallback(latestRemoteLots);
    }
  }

  await delay();
  return mergeRemoteLotsWithFallback(latestRemoteLots);
}

export async function getParkingLotById(lotId: string): Promise<ParkingLot | null> {
  const lots = await getParkingLots();
  const lot = lots.find((item) => item.id === lotId);
  return lot ? { ...lot } : null;
}

export async function getDashboardStats(): Promise<DashboardStats> {
  await ensureCameraStoreReady();

  if (shouldUseRemoteData()) {
    const remoteStats = await fetchRemoteDashboard();
    if (remoteStats) {
      return { ...remoteStats };
    }

    throwWhenApiStrict("Unable to load dashboard stats from API.");
  }

  await delay();
  const lotsSnapshot = buildDynamicLots();
  const totalCapacity = lotsSnapshot.reduce((sum, lot) => sum + lot.capacity, 0);
  const totalOccupied = lotsSnapshot.reduce((sum, lot) => sum + lot.occupied, 0);
  const occupancyPct = totalCapacity === 0 ? 0 : Math.round((totalOccupied / totalCapacity) * 100);
  const fullLots = lotsSnapshot.filter((lot) => lot.status === "full").length;
  const almostFullLots = lotsSnapshot.filter((lot) => lot.status === "almost_full").length;
  const offlineCameras = cameraStore.filter((camera) => camera.status === "offline").length;
  const alertBonus = scenarioConfig[activeScenario].alertBonus;
  const activeAlerts = Math.max(alertsMock.length + alertBonus, fullLots * 2 + almostFullLots + offlineCameras);
  const backendStatus = await fetchRemoteStatus();
  const bucket = getCurrentBucket();
  const mockedFps = Number((dashboardMock.systemFps + Math.sin(bucket * 0.21) * 0.6).toFixed(1));

  return {
    totalLots: lotsSnapshot.length,
    occupancyPct,
    activeAlerts,
    systemFps: backendStatus?.fps_estimate ?? mockedFps
  };
}

export async function getAlerts(): Promise<AlertItem[]> {
  if (shouldUseRemoteData()) {
    const remoteAlerts = await fetchRemoteAlerts();
    if (remoteAlerts) {
      return cloneAlerts(remoteAlerts);
    }

    throwWhenApiStrict("Unable to load alerts from API.");
  }

  await delay();
  return cloneAlerts(alertsMock);
}

export async function getCameras(): Promise<CameraItem[]> {
  await ensureCameraStoreReady();

  if (shouldUseRemoteData()) {
    const remoteCameras = await fetchRemoteCameras();
    if (remoteCameras) {
      cameraStore = cloneCameras(remoteCameras);
      await persistCameraStore();
      return cloneCameras(cameraStore);
    }

    throwWhenApiStrict("Unable to load cameras from API.");
  }

  await delay();
  return cloneCameras(cameraStore);
}

export async function createCamera(input: NewCameraInput): Promise<CameraItem> {
  await ensureCameraStoreReady();

  if (shouldUseRemoteData()) {
    const remoteCamera = await createRemoteCamera(input);
    if (remoteCamera) {
      cameraStore = [remoteCamera, ...cameraStore.filter((camera) => camera.id !== remoteCamera.id)];
      await persistCameraStore();
      return { ...remoteCamera };
    }

    throwWhenApiStrict("Unable to create camera via API.");
  }

  await delay(250);
  const timestamp = Date.now();
  const resolvedName = input.name.trim() || `Camera ${cameraStore.length + 1}`;
  const resolvedLocation = input.location.trim();

  const newCamera: CameraItem = {
    id: `cam-${timestamp.toString(36)}`,
    name: resolvedLocation ? `${resolvedName} - ${resolvedLocation}` : resolvedName,
    fps: 24,
    uptime: "100.0%",
    lastDetection: "Just now",
    status: "online",
    streamUrl: input.streamUrl.trim()
  };

  cameraStore = [newCamera, ...cameraStore];
  await persistCameraStore();
  return { ...newCamera };
}

export async function resetLocalCameraStore(): Promise<void> {
  cameraStore = cloneCameras(camerasMock);
  cameraStoreHydrationPromise = null;
  await removeJson(CAMERA_STORE_KEY);
}

export async function getUserProfile(): Promise<UserProfile> {
  if (shouldUseRemoteData()) {
    const remoteProfile = await fetchRemoteProfile();
    if (remoteProfile) {
      return { ...remoteProfile };
    }

    throwWhenApiStrict("Unable to load profile from API.");
  }

  await delay();
  return { ...profileMock };
}
