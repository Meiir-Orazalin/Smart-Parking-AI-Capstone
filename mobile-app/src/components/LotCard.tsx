import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppCard } from "./AppCard";
import { ProgressBar } from "./ProgressBar";
import { StatusPill } from "./StatusPill";
import { theme } from "../constants/theme";
import { ParkingLot } from "../types/models";

type LotCardProps = ParkingLot & {
  onPress?: () => void;
};

export function LotCard({
  name,
  occupied,
  capacity,
  status,
  distanceMi,
  lastUpdated,
  onPress
}: LotCardProps) {
  const updatedLabel = new Date(lastUpdated).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit"
  });

  const content = (
    <AppCard style={styles.card}>
      <View style={styles.headerRow}>
        <Text style={styles.name}>{name}</Text>
        <StatusPill label={status} />
      </View>

      <Text style={styles.occupancy}>
        {occupied}/{capacity}
      </Text>
      <ProgressBar value={occupied} max={capacity} status={status} />

      <View style={styles.metaRow}>
        <View style={styles.metaItem}>
          <Ionicons name="location-outline" size={14} color={theme.colors.textMuted} />
          <Text style={styles.metaText}>{distanceMi.toFixed(1)} mi</Text>
        </View>
        <View style={styles.metaItem}>
          <Ionicons name="time-outline" size={14} color={theme.colors.textMuted} />
          <Text style={styles.metaText}>Updated {updatedLabel}</Text>
        </View>
      </View>
    </AppCard>
  );

  if (!onPress) {
    return content;
  }

  return <Pressable onPress={onPress}>{content}</Pressable>;
}

const styles = StyleSheet.create({
  card: {
    gap: theme.spacing.sm
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: theme.spacing.sm
  },
  name: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
    flex: 1
  },
  occupancy: {
    ...theme.typography.bodyBold,
    color: theme.colors.textPrimary
  },
  metaRow: {
    marginTop: theme.spacing.xs,
    flexDirection: "row",
    gap: theme.spacing.md
  },
  metaItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing.xs
  },
  metaText: {
    ...theme.typography.caption,
    color: theme.colors.textMuted
  }
});
