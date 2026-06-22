import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { useAppData } from "../hooks/AppDataProvider";
import { colors, radii, spacing, typography } from "../theme";

export function EnvironmentBanner() {
  const { dataMode, apiBaseUrl, localOnlyData, preflightIssues } = useAppData();

  const hasIssues = preflightIssues.length > 0;
  if (!hasIssues && !localOnlyData) {
    return null;
  }

  return (
    <View style={styles.container}>
      {hasIssues ? (
        <View style={styles.issueCard}>
          <Text style={styles.issueTitle}>Config Check</Text>
          {preflightIssues.map((issue) => (
            <Text key={issue.code} style={styles.issueText}>
              - {issue.message}
            </Text>
          ))}
        </View>
      ) : null}

      {localOnlyData ? (
        <View style={styles.infoCard}>
          <Text style={styles.infoText}>
            Running in {dataMode.toUpperCase()} mode. Camera additions and alert workflow are local per device until
            backend persistence is connected. API base: {apiBaseUrl}
          </Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: spacing.xs
  },
  issueCard: {
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: "rgba(255, 93, 93, 0.45)",
    backgroundColor: "rgba(255, 93, 93, 0.12)",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm
  },
  issueTitle: {
    ...typography.caption,
    color: colors.danger,
    fontWeight: "700",
    marginBottom: 2
  },
  issueText: {
    ...typography.caption,
    color: colors.textPrimary
  },
  infoCard: {
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surfaceAlt,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm
  },
  infoText: {
    ...typography.caption,
    color: colors.textSecondary
  }
});
