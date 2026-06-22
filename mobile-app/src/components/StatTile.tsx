import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { AppCard } from "./AppCard";
import { colors, typography } from "../theme";

type StatTileProps = {
  label: string;
  value: string | number;
  accent?: string;
};

export function StatTile({ label, value, accent = colors.accent }: StatTileProps) {
  return (
    <AppCard style={styles.card}>
      <Text style={styles.label}>{label}</Text>
      <Text style={[styles.value, { color: accent }]}>{value}</Text>
    </AppCard>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    paddingVertical: 14,
    paddingHorizontal: 12
  },
  label: {
    ...typography.caption,
    color: colors.textSecondary
  },
  value: {
    ...typography.h3,
    marginTop: 4
  }
});
