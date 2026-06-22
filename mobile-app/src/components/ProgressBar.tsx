import React from "react";
import { StyleSheet, View } from "react-native";
import { theme } from "../constants/theme";
import { LotStatus } from "../types/models";

type ProgressBarProps = {
  value: number;
  max: number;
  status: LotStatus;
};

function getFillColor(status: LotStatus) {
  if (status === "full") return theme.colors.danger;
  if (status === "almost_full") return theme.colors.warning;
  return theme.colors.success;
}

export function ProgressBar({ value, max, status }: ProgressBarProps) {
  const percentage = max === 0 ? 0 : Math.max(0, Math.min(100, (value / max) * 100));

  return (
    <View style={styles.track}>
      <View style={[styles.fill, { width: `${percentage}%`, backgroundColor: getFillColor(status) }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  track: {
    height: 10,
    borderRadius: theme.radii.pill,
    overflow: "hidden",
    backgroundColor: theme.colors.cardAlt
  },
  fill: {
    height: "100%",
    borderRadius: theme.radii.pill
  }
});
