import { useAppData } from "./AppDataProvider";

export function useDashboard() {
  const {
    stats,
    alerts,
    cameras,
    profile,
    alertWorkflow,
    loading,
    refreshing,
    lastSyncedAt,
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
  } = useAppData();

  return {
    stats,
    alerts,
    cameras,
    profile,
    alertWorkflow,
    loading,
    refreshing,
    lastSyncedAt,
    error: dashboardError,
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
  };
}
