import React, { memo, useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  ListRenderItemInfo,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View
} from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { ScreenContainer } from "../../components/ScreenContainer";
import { SearchBar } from "../../components/SearchBar";
import { AppCard } from "../../components/AppCard";
import { PrimaryButton } from "../../components/PrimaryButton";
import { StatTile } from "../../components/StatTile";
import { StatusPill } from "../../components/StatusPill";
import { EnvironmentBanner } from "../../components/EnvironmentBanner";
import { useDashboard } from "../../hooks/useDashboard";
import { useLots } from "../../hooks/useLots";
import { useRoleSwitcher } from "../../hooks/useRoleSwitcher";
import { AlertItem, AlertWorkflowStatus, ParkingLot } from "../../types/models";
import { colors, radii, spacing, typography } from "../../theme";

const STALE_THRESHOLD_MS = 45_000;
const FRESHNESS_TICK_MS = 10_000;

function lotPriorityScore(lot: ParkingLot): number {
  const ratio = lot.capacity === 0 ? 0 : lot.occupied / lot.capacity;
  const normalizedName = lot.name.trim().toLowerCase();
  const mainLotWeight =
    lot.id === "main-lot" || normalizedName === "main lot" || normalizedName === "rit dubai dormitory" ? 1000 : 0;
  const statusWeight = lot.status === "full" ? 300 : lot.status === "almost_full" ? 200 : 100;
  return mainLotWeight + statusWeight + ratio * 100;
}

function formatFreshness(lastSyncedAt: string | null, refreshing: boolean, clockMs: number): string {
  if (refreshing) return "Refreshing data...";
  if (!lastSyncedAt) return "Not synced yet";

  const ageSec = Math.max(0, Math.round((clockMs - new Date(lastSyncedAt).getTime()) / 1000));
  if (ageSec < 5) return "Updated just now";
  if (ageSec < 60) return `Updated ${ageSec}s ago`;
  return `Updated ${Math.round(ageSec / 60)}m ago`;
}

function isStale(lastSyncedAt: string | null, clockMs: number): boolean {
  if (!lastSyncedAt) return true;
  return clockMs - new Date(lastSyncedAt).getTime() > STALE_THRESHOLD_MS;
}

function toWorkflowLabel(status: AlertWorkflowStatus): string {
  if (status === "new") return "New";
  if (status === "acknowledged") return "Acknowledged";
  if (status === "assigned") return "Assigned";
  return "Resolved";
}

function formatUpdatedLabel(iso: string): string {
  const timestamp = new Date(iso).getTime();
  const diffMinutes = Math.max(0, Math.round((Date.now() - timestamp) / 60000));
  if (diffMinutes < 1) return "Just now";
  return `${diffMinutes} min ago`;
}

function operatorActionForLot(lot: ParkingLot): string {
  if (lot.status === "full") {
    return "Activate overflow lane and update entry signage.";
  }
  if (lot.status === "almost_full") {
    return "Monitor inflow and pre-stage attendants for peak volume.";
  }
  return "Normal load. Keep standard monitoring cadence.";
}

type LotCardProps = {
  lot: ParkingLot;
  onPress: (lotId: string) => void;
};

const LotCard = memo(function LotCard({ lot, onPress }: LotCardProps) {
  const ratio = lot.capacity === 0 ? 0 : lot.occupied / lot.capacity;
  const trendIcon = ratio > 0.8 ? "trending-up" : "trending-down";
  const trendColor = ratio > 0.8 ? colors.warning : colors.success;

  return (
    <Pressable
      onPress={() => onPress(lot.id)}
      accessibilityRole="button"
      accessibilityLabel={`Open operator lot details for ${lot.name}`}
    >
      <AppCard style={styles.rowCard}>
        <View style={styles.rowTop}>
          <Text style={styles.lotName}>{lot.name}</Text>
          <StatusPill label={lot.status} />
        </View>
        <View style={styles.rowBottom}>
          <Text style={styles.occupancy}>
            {lot.occupied}/{lot.capacity} occupied
          </Text>
          <View style={styles.trend}>
            <MaterialCommunityIcons name={trendIcon} size={16} color={trendColor} />
            <Text style={[styles.trendText, { color: trendColor }]}>
              {ratio > 0.8 ? "Rising load" : "Stable flow"}
            </Text>
          </View>
        </View>
      </AppCard>
    </Pressable>
  );
});

type AlertCardProps = {
  alert: AlertItem;
  workflow: AlertWorkflowStatus;
  setAlertStatus: (alertId: string, status: AlertWorkflowStatus) => void;
};

const AlertCard = memo(function AlertCard({ alert, workflow, setAlertStatus }: AlertCardProps) {
  const canAcknowledge = workflow === "new";
  const canAssign = workflow === "acknowledged";
  const canResolve = workflow === "acknowledged" || workflow === "assigned";

  return (
    <AppCard style={styles.alertCard}>
      <View style={styles.rowTop}>
        <Text style={styles.alertTitle}>{alert.title}</Text>
        <StatusPill label={alert.severity} />
      </View>

      <Text style={styles.alertMeta}>
        {alert.location} - {alert.timeAgo}
      </Text>

      <View style={styles.workflowRow}>
        <Text
          style={[
            styles.workflowText,
            workflow === "new" && styles.workflowNew,
            workflow === "acknowledged" && styles.workflowAcknowledged,
            workflow === "assigned" && styles.workflowAssigned,
            workflow === "resolved" && styles.workflowResolved
          ]}
        >
          {toWorkflowLabel(workflow)}
        </Text>
      </View>

      <View style={styles.alertActions}>
        <Pressable
          onPress={() => setAlertStatus(alert.id, "acknowledged")}
          disabled={!canAcknowledge}
          style={[styles.alertActionButton, !canAcknowledge && styles.alertActionButtonDisabled]}
        >
          <Text style={[styles.alertActionText, !canAcknowledge && styles.alertActionTextDisabled]}>
            Acknowledge
          </Text>
        </Pressable>

        <Pressable
          onPress={() => setAlertStatus(alert.id, "assigned")}
          disabled={!canAssign}
          style={[styles.alertActionButton, !canAssign && styles.alertActionButtonDisabled]}
        >
          <Text style={[styles.alertActionText, !canAssign && styles.alertActionTextDisabled]}>
            Assign
          </Text>
        </Pressable>

        <Pressable
          onPress={() => setAlertStatus(alert.id, "resolved")}
          disabled={!canResolve}
          style={[styles.alertActionButton, !canResolve && styles.alertActionButtonDisabled]}
        >
          <Text style={[styles.alertActionText, !canResolve && styles.alertActionTextDisabled]}>
            Resolve
          </Text>
        </Pressable>
      </View>
    </AppCard>
  );
});

export function DashboardScreen() {
  const [query, setQuery] = useState("");
  const [clockMs, setClockMs] = useState(() => Date.now());
  const [selectedLotId, setSelectedLotId] = useState<string | null>(null);
  const [showResolved, setShowResolved] = useState(false);

  const {
    stats,
    alerts,
    cameras,
    alertWorkflow,
    loading: dashboardLoading,
    refreshing: dashboardRefreshing,
    error: dashboardError,
    lastSyncedAt,
    refresh,
    setAlertStatus
  } = useDashboard();

  const {
    lots,
    loading: lotsLoading,
    refreshing: lotsRefreshing,
    error: lotsError
  } = useLots();

  const { switchRole } = useRoleSwitcher();

  useEffect(() => {
    const timer = setInterval(() => {
      setClockMs(Date.now());
    }, FRESHNESS_TICK_MS);

    return () => clearInterval(timer);
  }, []);

  const filteredLots = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const subset = normalized
      ? lots.filter((lot) => lot.name.toLowerCase().includes(normalized))
      : lots;

    return [...subset].sort((a, b) => lotPriorityScore(b) - lotPriorityScore(a));
  }, [lots, query]);

  const attentionCount = useMemo(
    () => lots.filter((lot) => lot.status === "full" || lot.status === "almost_full").length,
    [lots]
  );

  const selectedLot = useMemo(
    () => lots.find((lot) => lot.id === selectedLotId) ?? null,
    [lots, selectedLotId]
  );

  const lotLinkedAlerts = useMemo(() => {
    if (!selectedLot) return 0;
    const target = selectedLot.name.toLowerCase();
    return alerts.filter((alert) => alert.location.toLowerCase().includes(target)).length;
  }, [alerts, selectedLot]);

  const lotLinkedCameras = useMemo(() => {
    if (!selectedLot) return 0;
    const lotKey = selectedLot.name.split(" ")[0].toLowerCase();
    return cameras.filter((camera) => camera.name.toLowerCase().includes(lotKey)).length;
  }, [cameras, selectedLot]);

  const freshnessLabel = formatFreshness(lastSyncedAt, dashboardRefreshing || lotsRefreshing, clockMs);
  const stale = isStale(lastSyncedAt, clockMs);

  const getAlertStatus = useCallback(
    (alertId: string): AlertWorkflowStatus => alertWorkflow[alertId] ?? "new",
    [alertWorkflow]
  );

  const unresolvedAlerts = useMemo(
    () => alerts.filter((alert) => getAlertStatus(alert.id) !== "resolved"),
    [alerts, getAlertStatus]
  );

  const resolvedAlerts = useMemo(
    () => alerts.filter((alert) => getAlertStatus(alert.id) === "resolved"),
    [alerts, getAlertStatus]
  );

  const renderLot = useCallback(
    ({ item }: ListRenderItemInfo<ParkingLot>) => (
      <LotCard lot={item} onPress={setSelectedLotId} />
    ),
    []
  );

  const header = (
    <View style={styles.header}>
      <Text style={styles.title}>Operations Dashboard</Text>
      <Text style={styles.subtitle}>System health and occupancy intelligence</Text>
      <EnvironmentBanner />

      <View style={styles.topActions}>
        <PrimaryButton label="Switch Role" variant="outline" onPress={switchRole} style={styles.flexButton} />
        <PrimaryButton label="Refresh" variant="outline" onPress={() => void refresh()} style={styles.flexButton} />
      </View>

      <View style={styles.syncRow}>
        <Text style={[styles.syncText, stale && styles.staleText]}>{freshnessLabel}</Text>
        {stale ? (
          <View style={styles.staleBadge}>
            <Text style={styles.staleBadgeText}>Stale</Text>
          </View>
        ) : null}
      </View>

      <View style={styles.statRow}>
        <StatTile label="Total Lots" value={stats?.totalLots ?? "--"} />
        <StatTile
          label="Occupancy"
          value={stats ? `${stats.occupancyPct}%` : "--"}
          accent={colors.warning}
        />
        <StatTile
          label="Active Alerts"
          value={stats?.activeAlerts ?? "--"}
          accent={colors.danger}
        />
      </View>

      <SearchBar value={query} onChangeText={setQuery} placeholder="Search lots..." />

      {dashboardLoading || lotsLoading ? (
        <View style={styles.loadingRow}>
          <ActivityIndicator size="small" color={colors.accent} />
          <Text style={styles.loadingText}>Loading operations data...</Text>
        </View>
      ) : null}

      {dashboardError ? <Text style={styles.errorText}>{dashboardError}</Text> : null}
      {lotsError ? <Text style={styles.errorText}>{lotsError}</Text> : null}

      <View style={styles.sectionHeaderRow}>
        <Text style={styles.sectionTitle}>Lots - Needs Attention First</Text>
        <Text style={styles.sectionHint}>{attentionCount} flagged</Text>
      </View>
    </View>
  );

  const footer = (
    <View style={styles.footer}>
      <View style={styles.sectionHeaderRow}>
        <Text style={styles.sectionTitle}>Recent Alerts</Text>
        <Text style={styles.sectionHint}>{unresolvedAlerts.length} active</Text>
      </View>

      <View style={styles.list}>
        {unresolvedAlerts.length === 0 ? (
          <AppCard>
            <Text style={styles.emptyStateText}>No active alerts. All clear.</Text>
          </AppCard>
        ) : (
          unresolvedAlerts.slice(0, 4).map((item) => (
            <AlertCard
              key={item.id}
              alert={item}
              workflow={getAlertStatus(item.id)}
              setAlertStatus={setAlertStatus}
            />
          ))
        )}
      </View>

      {resolvedAlerts.length > 0 ? (
        <View style={styles.resolvedSection}>
          <Pressable style={styles.resolvedHeader} onPress={() => setShowResolved((current) => !current)}>
            <Text style={styles.resolvedTitle}>Resolved Alerts ({resolvedAlerts.length})</Text>
            <MaterialCommunityIcons
              name={showResolved ? "chevron-up" : "chevron-down"}
              size={18}
              color={colors.textMuted}
            />
          </Pressable>

          {showResolved ? (
            <View style={styles.list}>
              {resolvedAlerts.map((item) => (
                <AppCard key={`resolved-${item.id}`} style={styles.alertCard}>
                  <View style={styles.rowTop}>
                    <Text style={styles.alertTitle}>{item.title}</Text>
                    <StatusPill label={item.severity} />
                  </View>
                  <Text style={styles.alertMeta}>
                    {item.location} - {item.timeAgo}
                  </Text>
                  <View style={styles.alertActions}>
                    <Pressable
                      onPress={() => setAlertStatus(item.id, "acknowledged")}
                      style={styles.alertActionButton}
                    >
                      <Text style={styles.alertActionText}>Re-open</Text>
                    </Pressable>
                  </View>
                </AppCard>
              ))}
            </View>
          ) : null}
        </View>
      ) : null}
    </View>
  );

  return (
    <>
      <ScreenContainer scroll={false}>
        <View style={styles.page}>
          <FlatList
            data={filteredLots}
            renderItem={renderLot}
            keyExtractor={(item) => item.id}
            ListHeaderComponent={header}
            ListFooterComponent={footer}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
            initialNumToRender={8}
            maxToRenderPerBatch={10}
            windowSize={9}
            removeClippedSubviews
            refreshing={dashboardRefreshing || lotsRefreshing}
            onRefresh={() => void refresh()}
            keyboardShouldPersistTaps="handled"
          />
        </View>
      </ScreenContainer>

      <Modal
        visible={selectedLot !== null}
        transparent
        animationType="slide"
        onRequestClose={() => setSelectedLotId(null)}
      >
        <View style={styles.modalRoot}>
          <Pressable style={styles.modalBackdrop} onPress={() => setSelectedLotId(null)} />

          <View style={styles.modalSheet}>
            {selectedLot ? (
              <>
                <View style={styles.modalHeaderRow}>
                  <Text style={styles.modalTitle}>{selectedLot.name}</Text>
                  <StatusPill label={selectedLot.status} />
                </View>

                <Text style={styles.modalSubtitle}>Operator lot details</Text>

                <View style={styles.modalDetailsList}>
                  <View style={styles.modalDetailRow}>
                    <Text style={styles.modalDetailLabel}>Occupancy</Text>
                    <Text style={styles.modalDetailValue}>
                      {selectedLot.capacity === 0
                        ? "0%"
                        : `${Math.round((selectedLot.occupied / selectedLot.capacity) * 100)}%`}
                    </Text>
                  </View>
                  <View style={styles.modalDetailRow}>
                    <Text style={styles.modalDetailLabel}>Spaces</Text>
                    <Text style={styles.modalDetailValue}>
                      {selectedLot.occupied}/{selectedLot.capacity} occupied
                    </Text>
                  </View>
                  <View style={styles.modalDetailRow}>
                    <Text style={styles.modalDetailLabel}>Free</Text>
                    <Text style={styles.modalDetailValue}>
                      {Math.max(0, selectedLot.capacity - selectedLot.occupied)} spaces
                    </Text>
                  </View>
                  <View style={styles.modalDetailRow}>
                    <Text style={styles.modalDetailLabel}>Distance</Text>
                    <Text style={styles.modalDetailValue}>{selectedLot.distanceMi.toFixed(1)} mi</Text>
                  </View>
                  <View style={styles.modalDetailRow}>
                    <Text style={styles.modalDetailLabel}>Last Updated</Text>
                    <Text style={styles.modalDetailValue}>{formatUpdatedLabel(selectedLot.lastUpdated)}</Text>
                  </View>
                  <View style={styles.modalDetailRow}>
                    <Text style={styles.modalDetailLabel}>Linked Alerts</Text>
                    <Text style={styles.modalDetailValue}>{lotLinkedAlerts}</Text>
                  </View>
                  <View style={styles.modalDetailRow}>
                    <Text style={styles.modalDetailLabel}>Linked Cameras</Text>
                    <Text style={styles.modalDetailValue}>{lotLinkedCameras}</Text>
                  </View>
                </View>

                <AppCard style={styles.operatorNoteCard}>
                  <Text style={styles.operatorNoteTitle}>Suggested Operator Action</Text>
                  <Text style={styles.operatorNoteText}>{operatorActionForLot(selectedLot)}</Text>
                </AppCard>

                <PrimaryButton label="Close" variant="outline" onPress={() => setSelectedLotId(null)} />
              </>
            ) : null}
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  page: {
    flex: 1
  },
  listContent: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xl,
    gap: spacing.sm
  },
  header: {
    gap: spacing.md
  },
  footer: {
    gap: spacing.sm
  },
  title: {
    ...typography.h1,
    color: colors.textPrimary
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary
  },
  topActions: {
    flexDirection: "row",
    gap: spacing.sm
  },
  flexButton: {
    flex: 1
  },
  syncRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm
  },
  syncText: {
    ...typography.caption,
    color: colors.textMuted
  },
  staleText: {
    color: colors.warning
  },
  staleBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: radii.pill,
    backgroundColor: "rgba(247, 183, 49, 0.18)",
    borderWidth: 1,
    borderColor: "rgba(247, 183, 49, 0.35)"
  },
  staleBadgeText: {
    ...typography.caption,
    color: colors.warning,
    fontWeight: "700"
  },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm
  },
  loadingText: {
    ...typography.caption,
    color: colors.textMuted
  },
  errorText: {
    ...typography.caption,
    color: colors.danger
  },
  statRow: {
    flexDirection: "row",
    gap: spacing.sm
  },
  sectionHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center"
  },
  sectionHint: {
    ...typography.caption,
    color: colors.warning
  },
  list: {
    gap: spacing.sm
  },
  rowCard: {
    gap: spacing.xs
  },
  rowTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.sm
  },
  rowBottom: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center"
  },
  lotName: {
    ...typography.bodyBold,
    color: colors.textPrimary,
    flex: 1
  },
  occupancy: {
    ...typography.caption,
    color: colors.textSecondary
  },
  trend: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs
  },
  trendText: {
    ...typography.caption
  },
  sectionTitle: {
    ...typography.h3,
    color: colors.textPrimary,
    marginTop: spacing.sm
  },
  alertCard: {
    gap: spacing.xs
  },
  alertTitle: {
    ...typography.bodyBold,
    color: colors.textPrimary,
    flex: 1
  },
  alertMeta: {
    ...typography.caption,
    color: colors.textMuted
  },
  workflowRow: {
    marginTop: 2
  },
  workflowText: {
    ...typography.caption,
    fontWeight: "700"
  },
  workflowNew: {
    color: colors.textMuted
  },
  workflowAcknowledged: {
    color: colors.warning
  },
  workflowAssigned: {
    color: colors.accent
  },
  workflowResolved: {
    color: colors.success
  },
  alertActions: {
    flexDirection: "row",
    gap: spacing.xs,
    marginTop: spacing.xs
  },
  alertActionButton: {
    flex: 1,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surfaceAlt,
    paddingVertical: 8,
    alignItems: "center",
    justifyContent: "center"
  },
  alertActionButtonDisabled: {
    opacity: 0.45
  },
  alertActionText: {
    ...typography.caption,
    color: colors.textPrimary,
    fontWeight: "700"
  },
  alertActionTextDisabled: {
    color: colors.textMuted
  },
  emptyStateText: {
    ...typography.caption,
    color: colors.textMuted
  },
  resolvedSection: {
    marginTop: spacing.sm,
    gap: spacing.sm
  },
  resolvedHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between"
  },
  resolvedTitle: {
    ...typography.bodyBold,
    color: colors.textSecondary
  },
  modalRoot: {
    flex: 1,
    justifyContent: "flex-end"
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: colors.overlay
  },
  modalSheet: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: radii.lg,
    borderTopRightRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.md,
    maxHeight: "82%"
  },
  modalHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.sm
  },
  modalTitle: {
    ...typography.h2,
    color: colors.textPrimary,
    flex: 1
  },
  modalSubtitle: {
    ...typography.caption,
    color: colors.textSecondary
  },
  modalDetailsList: {
    gap: spacing.xs
  },
  modalDetailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.sm,
    paddingVertical: 2
  },
  modalDetailLabel: {
    ...typography.caption,
    color: colors.textSecondary
  },
  modalDetailValue: {
    ...typography.bodyBold,
    color: colors.textPrimary
  },
  operatorNoteCard: {
    backgroundColor: colors.surfaceAlt,
    gap: spacing.xs
  },
  operatorNoteTitle: {
    ...typography.caption,
    color: colors.textSecondary,
    textTransform: "uppercase",
    letterSpacing: 0.6
  },
  operatorNoteText: {
    ...typography.body,
    color: colors.textPrimary
  }
});
