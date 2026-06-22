import React from "react";
import { StyleSheet, Text } from "react-native";
import { AppCard } from "./AppCard";
import { theme } from "../constants/theme";

type StatCardProps = {
  title: string;
  value: string;
  delta: string;
};

export function StatCard({ title, value, delta }: StatCardProps) {
  return (
    <AppCard style={styles.card}>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.value}>{value}</Text>
      <Text style={styles.delta}>{delta}</Text>
    </AppCard>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.md,
    gap: theme.spacing.xs
  },
  title: {
    ...theme.typography.caption,
    color: theme.colors.textSecondary
  },
  value: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary
  },
  delta: {
    ...theme.typography.caption,
    color: theme.colors.textMuted
  }
});
