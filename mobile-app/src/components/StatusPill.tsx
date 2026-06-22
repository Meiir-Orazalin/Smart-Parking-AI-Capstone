import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { theme } from "../constants/theme";
import { AlertSeverity, CameraStatus, LotStatus } from "../types/models";

type StatusPillProps = {
  label: LotStatus | AlertSeverity | CameraStatus;
};

function getColors(label: StatusPillProps["label"]) {
  if (label === "available" || label === "low" || label === "online") {
    return {
      text: theme.colors.success,
      bg: "rgba(46, 204, 113, 0.2)"
    };
  }

  if (label === "almost_full" || label === "medium") {
    return {
      text: theme.colors.warning,
      bg: "rgba(247, 183, 49, 0.2)"
    };
  }

  return {
    text: theme.colors.danger,
    bg: "rgba(255, 93, 93, 0.2)"
  };
}

function toDisplayLabel(label: StatusPillProps["label"]): string {
  if (label === "almost_full") return "Almost Full";
  if (label === "available") return "Available";
  if (label === "full") return "Full";
  if (label === "low") return "Low";
  if (label === "medium") return "Medium";
  if (label === "high") return "High";
  if (label === "online") return "Online";
  return "Offline";
}

export function StatusPill({ label }: StatusPillProps) {
  const palette = getColors(label);
  return (
    <View style={[styles.container, { backgroundColor: palette.bg }]}>
      <Text style={[styles.text, { color: palette.text }]}>{toDisplayLabel(label)}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: theme.radii.pill,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.xs
  },
  text: {
    ...theme.typography.caption,
    fontWeight: "700"
  }
});
