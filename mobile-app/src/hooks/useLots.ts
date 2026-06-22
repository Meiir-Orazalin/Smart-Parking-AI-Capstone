import { useAppData } from "./AppDataProvider";

export function useLots() {
  const { lots, loading, refreshing, lastSyncedAt, lotsError, refresh } = useAppData();

  return {
    lots,
    loading,
    refreshing,
    lastSyncedAt,
    error: lotsError,
    refresh
  };
}
