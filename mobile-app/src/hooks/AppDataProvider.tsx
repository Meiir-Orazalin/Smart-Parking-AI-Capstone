import React, {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  createCamera,
  getApiBaseUrl,
  getAlerts,
  getCameras,
  getDashboardStats,
  getMockScenario,
  getParkingLots,
  getRuntimePreflightIssues,
  getUserProfile,
  getDataMode,
  isLocalOnlyDataMode,
  listMockScenarios,
  resetLocalCameraStore,
  RuntimePreflightIssue,
  setMockScenario
} from "../services/api";
import { readJson, removeJson, writeJson } from "../services/storage";
import {
  AlertItem,
  AlertWorkflowStatus,
  CameraItem,
  DashboardStats,
  MockScenario,
  NewCameraInput,
  ParkingLot,
  UserProfile
} from "../types/models";

const FAST_REFRESH_INTERVAL_MS = 8000;
const STANDARD_REFRESH_INTERVAL_MS = 20000;
const PROFILE_REFRESH_INTERVAL_MS = 60000;
const ALERT_WORKFLOW_KEY = "smartpark:alert-workflow:v1";

type DashboardFailureKey = "stats" | "alerts" | "cameras" | "profile";
type DashboardFailureState = Record<DashboardFailureKey, boolean>;

const initialDashboardFailures = (): DashboardFailureState => ({
  stats: false,
  alerts: false,
  cameras: false,
  profile: false
});

type AppDataContextValue = {
  lots: ParkingLot[];
  stats: DashboardStats | null;
  alerts: AlertItem[];
  cameras: CameraItem[];
  profile: UserProfile | null;
  alertWorkflow: Record<string, AlertWorkflowStatus>;
  loading: boolean;
  refreshing: boolean;
  lastSyncedAt: string | null;
  lotsError: string | null;
  dashboardError: string | null;
  scenario: MockScenario;
  scenarios: MockScenario[];
  dataMode: ReturnType<typeof getDataMode>;
  apiBaseUrl: string;
  localOnlyData: boolean;
  preflightIssues: RuntimePreflightIssue[];
  refresh: () => Promise<void>;
  setScenario: (scenario: MockScenario) => Promise<void>;
  addCamera: (input: NewCameraInput) => Promise<CameraItem>;
  setAlertStatus: (alertId: string, status: AlertWorkflowStatus) => void;
  resetLocalDemoData: () => Promise<void>;
};

const AppDataContext = createContext<AppDataContextValue | undefined>(undefined);

export function AppDataProvider({ children }: { children: ReactNode }) {
  const [lots, setLots] = useState<ParkingLot[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [cameras, setCameras] = useState<CameraItem[]>([]);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [alertWorkflow, setAlertWorkflow] = useState<Record<string, AlertWorkflowStatus>>({});
  const [workflowHydrated, setWorkflowHydrated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastSyncedAt, setLastSyncedAt] = useState<string | null>(null);
  const [lotsError, setLotsError] = useState<string | null>(null);
  const [dashboardError, setDashboardError] = useState<string | null>(null);
  const [scenario, setScenarioState] = useState<MockScenario>(() => getMockScenario());
  const dataMode = useMemo(() => getDataMode(), []);
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);
  const localOnlyData = useMemo(() => isLocalOnlyDataMode(), []);
  const preflightIssues = useMemo(() => getRuntimePreflightIssues(), []);
  const scenarios = useMemo(() => listMockScenarios(), []);
  const mountedRef = useRef(true);
  const workflowPersistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dashboardFailuresRef = useRef<DashboardFailureState>(initialDashboardFailures());
  const fullRefreshInFlightRef = useRef(false);
  const fastPollInFlightRef = useRef(false);
  const standardPollInFlightRef = useRef(false);
  const profilePollInFlightRef = useRef(false);

  const markSynced = useCallback(() => {
    setLastSyncedAt(new Date().toISOString());
  }, []);

  const setDashboardFailures = useCallback((next: Partial<DashboardFailureState>) => {
    dashboardFailuresRef.current = {
      ...dashboardFailuresRef.current,
      ...next
    };

    const hasFailure = Object.values(dashboardFailuresRef.current).some(Boolean);
    setDashboardError(hasFailure ? "Failed to load dashboard data." : null);
  }, []);

  useEffect(() => {
    let active = true;

    const hydrateWorkflow = async () => {
      const persisted = await readJson<Record<string, AlertWorkflowStatus>>(ALERT_WORKFLOW_KEY, {});
      if (!active) return;
      setAlertWorkflow(persisted);
      setWorkflowHydrated(true);
    };

    void hydrateWorkflow();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!workflowHydrated) return;
    if (workflowPersistTimerRef.current) {
      clearTimeout(workflowPersistTimerRef.current);
    }

    workflowPersistTimerRef.current = setTimeout(() => {
      void writeJson(ALERT_WORKFLOW_KEY, alertWorkflow);
    }, 350);

    return () => {
      if (workflowPersistTimerRef.current) {
        clearTimeout(workflowPersistTimerRef.current);
        workflowPersistTimerRef.current = null;
      }
    };
  }, [alertWorkflow, workflowHydrated]);

  const loadAll = useCallback(async (mode: "initial" | "manual" | "silent" = "silent") => {
    if (fullRefreshInFlightRef.current) {
      return;
    }
    fullRefreshInFlightRef.current = true;

    if (mode === "initial") {
      setLoading(true);
    } else if (mode === "manual") {
      setRefreshing(true);
    }

    try {
      const [lotsResult, statsResult, alertsResult, camerasResult, profileResult] = await Promise.allSettled([
        getParkingLots(),
        getDashboardStats(),
        getAlerts(),
        getCameras(),
        getUserProfile()
      ]);

      if (!mountedRef.current) {
        return;
      }

      let synced = false;

      if (lotsResult.status === "fulfilled") {
        setLots(lotsResult.value);
        setLotsError(null);
        synced = true;
      } else {
        setLotsError("Failed to load parking lots.");
      }

      if (statsResult.status === "fulfilled") {
        setStats(statsResult.value);
        synced = true;
      }

      if (alertsResult.status === "fulfilled") {
        setAlerts(alertsResult.value);
        synced = true;
      }

      if (camerasResult.status === "fulfilled") {
        setCameras(camerasResult.value);
        synced = true;
      }

      if (profileResult.status === "fulfilled") {
        setProfile(profileResult.value);
        synced = true;
      }

      setDashboardFailures({
        stats: statsResult.status !== "fulfilled",
        alerts: alertsResult.status !== "fulfilled",
        cameras: camerasResult.status !== "fulfilled",
        profile: profileResult.status !== "fulfilled"
      });

      if (synced) {
        markSynced();
      }
    } finally {
      fullRefreshInFlightRef.current = false;
      if (!mountedRef.current) {
        return;
      }
      if (mode === "initial") {
        setLoading(false);
      }
      if (mode === "manual") {
        setRefreshing(false);
      }
    }
  }, [markSynced, setDashboardFailures]);

  const refreshLotsAndStats = useCallback(async () => {
    if (fullRefreshInFlightRef.current || fastPollInFlightRef.current) {
      return;
    }
    fastPollInFlightRef.current = true;

    try {
      const [lotsResult, statsResult] = await Promise.allSettled([getParkingLots(), getDashboardStats()]);
      if (!mountedRef.current) {
        return;
      }

      let synced = false;

      if (lotsResult.status === "fulfilled") {
        setLots(lotsResult.value);
        setLotsError(null);
        synced = true;
      } else {
        setLotsError("Failed to load parking lots.");
      }

      if (statsResult.status === "fulfilled") {
        setStats(statsResult.value);
        synced = true;
      }

      setDashboardFailures({
        stats: statsResult.status !== "fulfilled"
      });

      if (synced) {
        markSynced();
      }
    } finally {
      fastPollInFlightRef.current = false;
    }
  }, [markSynced, setDashboardFailures]);

  const refreshAlertsAndCameras = useCallback(async () => {
    if (fullRefreshInFlightRef.current || standardPollInFlightRef.current) {
      return;
    }
    standardPollInFlightRef.current = true;

    try {
      const [alertsResult, camerasResult] = await Promise.allSettled([getAlerts(), getCameras()]);
      if (!mountedRef.current) {
        return;
      }

      let synced = false;

      if (alertsResult.status === "fulfilled") {
        setAlerts(alertsResult.value);
        synced = true;
      }

      if (camerasResult.status === "fulfilled") {
        setCameras(camerasResult.value);
        synced = true;
      }

      setDashboardFailures({
        alerts: alertsResult.status !== "fulfilled",
        cameras: camerasResult.status !== "fulfilled"
      });

      if (synced) {
        markSynced();
      }
    } finally {
      standardPollInFlightRef.current = false;
    }
  }, [markSynced, setDashboardFailures]);

  const refreshProfile = useCallback(async () => {
    if (fullRefreshInFlightRef.current || profilePollInFlightRef.current) {
      return;
    }
    profilePollInFlightRef.current = true;

    try {
      const profileResult = await getUserProfile();
      if (!mountedRef.current) {
        return;
      }
      setProfile(profileResult);
      setDashboardFailures({
        profile: false
      });
      markSynced();
    } catch {
      if (!mountedRef.current) {
        return;
      }
      setDashboardFailures({
        profile: true
      });
    } finally {
      profilePollInFlightRef.current = false;
    }
  }, [markSynced, setDashboardFailures]);

  useEffect(() => {
    mountedRef.current = true;
    void loadAll("initial");

    const fastTimer = setInterval(() => {
      void refreshLotsAndStats();
    }, FAST_REFRESH_INTERVAL_MS);
    const standardTimer = setInterval(() => {
      void refreshAlertsAndCameras();
    }, STANDARD_REFRESH_INTERVAL_MS);
    const profileTimer = setInterval(() => {
      void refreshProfile();
    }, PROFILE_REFRESH_INTERVAL_MS);

    return () => {
      mountedRef.current = false;
      clearInterval(fastTimer);
      clearInterval(standardTimer);
      clearInterval(profileTimer);
    };
  }, [loadAll, refreshAlertsAndCameras, refreshLotsAndStats, refreshProfile]);

  const refresh = useCallback(async () => {
    await loadAll("manual");
  }, [loadAll]);

  const setScenario = useCallback(
    async (nextScenario: MockScenario) => {
      setMockScenario(nextScenario);
      setScenarioState(nextScenario);
      await loadAll("manual");
    },
    [loadAll]
  );

  const addCamera = useCallback(async (input: NewCameraInput) => {
    const created = await createCamera(input);

    if (mountedRef.current) {
      setCameras((current) => [created, ...current]);
      setLastSyncedAt(new Date().toISOString());
    }

    return created;
  }, []);

  const setAlertStatus = useCallback((alertId: string, status: AlertWorkflowStatus) => {
    setAlertWorkflow((current) => ({
      ...current,
      [alertId]: status
    }));
  }, []);

  const resetLocalDemoData = useCallback(async () => {
    await resetLocalCameraStore();
    await removeJson(ALERT_WORKFLOW_KEY);

    if (mountedRef.current) {
      setAlertWorkflow({});
      setWorkflowHydrated(true);
    }

    await loadAll("manual");
  }, [loadAll]);

  const value = useMemo<AppDataContextValue>(
    () => ({
      lots,
      stats,
      alerts,
      cameras,
      profile,
      alertWorkflow,
      loading,
      refreshing,
      lastSyncedAt,
      lotsError,
      dashboardError,
      scenario,
      scenarios,
      dataMode,
      apiBaseUrl,
      localOnlyData,
      preflightIssues,
      refresh,
      setScenario,
      addCamera,
      setAlertStatus,
      resetLocalDemoData
    }),
    [
      lots,
      stats,
      alerts,
      cameras,
      profile,
      alertWorkflow,
      loading,
      refreshing,
      lastSyncedAt,
      lotsError,
      dashboardError,
      scenario,
      scenarios,
      dataMode,
      apiBaseUrl,
      localOnlyData,
      preflightIssues,
      refresh,
      setScenario,
      addCamera,
      setAlertStatus,
      resetLocalDemoData
    ]
  );

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
}

export function useAppData() {
  const context = useContext(AppDataContext);
  if (!context) {
    throw new Error("useAppData must be used within an AppDataProvider.");
  }
  return context;
}
