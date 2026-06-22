import React, { useMemo } from "react";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { DriverLotsStackParamList } from "../../navigation/types";
import { useLots } from "../../hooks/useLots";
import { ScreenContainer } from "../../components/ScreenContainer";
import { StatTile } from "../../components/StatTile";
import { AppCard } from "../../components/AppCard";
import { BusynessChartCard } from "../../components/BusynessChartCard";
import { colors, radii, spacing, typography } from "../../theme";

type Props = NativeStackScreenProps<DriverLotsStackParamList, "LotDetails">;

function formatUpdatedLabel(iso: string) {
  const timestamp = new Date(iso).getTime();
  const diffMinutes = Math.max(0, Math.round((Date.now() - timestamp) / 60000));
  return `${diffMinutes} min ago`;
}

function formatStatus(status: string) {
  return status
    .split("_")
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

export function LotDetailsScreen({ route }: Props) {
  const { lots, loading, error } = useLots();
  const lot = useMemo(
    () => lots.find((item) => item.id === route.params.lotId),
    [lots, route.params.lotId]
  );

  if (loading) {
    return (
      <ScreenContainer>
        <View style={styles.loadingRow}>
          <ActivityIndicator size="small" color={colors.accent} />
          <Text style={styles.loadingText}>Loading lot details...</Text>
        </View>
      </ScreenContainer>
    );
  }

  if (error) {
    return (
      <ScreenContainer>
        <Text style={styles.errorText}>{error}</Text>
      </ScreenContainer>
    );
  }

  if (!lot) {
    return (
      <ScreenContainer>
        <Text style={styles.title}>Lot not found</Text>
      </ScreenContainer>
    );
  }

  const free = Math.max(0, lot.available ?? lot.capacity - lot.occupied);
  const unknown = Math.max(0, lot.unsure ?? Math.round(lot.capacity * 0.03));
  const occupancyPct = lot.capacity === 0 ? 0 : Math.round((lot.occupied / lot.capacity) * 100);

  return (
    <ScreenContainer>
      <Text style={styles.title}>{lot.name}</Text>
      <Text style={styles.subtitle}>Live lot composition and detailed parking insight</Text>

      <View style={styles.statsRow}>
        <StatTile label="Free" value={free} accent={colors.success} />
        <StatTile label="Occupied" value={lot.occupied} accent={colors.warning} />
        <StatTile label="Unknown" value={unknown} accent={colors.textMuted} />
      </View>

      <AppCard style={styles.mapPanel}>
        <Ionicons name="map-outline" size={28} color={colors.textMuted} />
        <Text style={styles.mapTitle}>Map Preview</Text>
        <Text style={styles.mapHint}>Interactive map integration is coming next.</Text>
      </AppCard>

      <AppCard style={styles.detailsCard}>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Status</Text>
          <Text style={styles.detailValue}>{formatStatus(lot.status)}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Occupancy</Text>
          <Text style={styles.detailValue}>{occupancyPct}%</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Capacity</Text>
          <Text style={styles.detailValue}>{lot.capacity} spaces</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Distance</Text>
          <Text style={styles.detailValue}>{lot.distanceMi.toFixed(1)} mi</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Last Updated</Text>
          <Text style={styles.detailValue}>{formatUpdatedLabel(lot.lastUpdated)}</Text>
        </View>
      </AppCard>

      <BusynessChartCard key={lot.id} lot={lot} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  title: {
    ...typography.h1,
    color: colors.textPrimary
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary
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
    ...typography.body,
    color: colors.danger
  },
  statsRow: {
    flexDirection: "row",
    gap: spacing.sm
  },
  mapPanel: {
    minHeight: 190,
    borderRadius: radii.lg,
    backgroundColor: colors.surfaceAlt,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xs
  },
  mapTitle: {
    ...typography.h3,
    color: colors.textPrimary
  },
  mapHint: {
    ...typography.caption,
    color: colors.textSecondary
  },
  detailsCard: {
    gap: spacing.xs
  },
  detailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.md,
    paddingVertical: 2
  },
  detailLabel: {
    ...typography.caption,
    color: colors.textSecondary
  },
  detailValue: {
    ...typography.bodyBold,
    color: colors.textPrimary
  }
});
