import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { theme } from "../constants/theme";

type SectionHeaderProps = {
  title: string;
  subtitle?: string;
  rightText?: string;
};

export function SectionHeader({ title, subtitle, rightText }: SectionHeaderProps) {
  return (
    <View style={styles.wrapper}>
      <View style={styles.left}>
        <Text style={styles.title}>{title}</Text>
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>
      {rightText ? <Text style={styles.rightText}>{rightText}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-end",
    gap: theme.spacing.md
  },
  left: {
    flex: 1
  },
  title: {
    ...theme.typography.h1,
    color: theme.colors.textPrimary
  },
  subtitle: {
    ...theme.typography.body,
    color: theme.colors.textSecondary,
    marginTop: theme.spacing.xs
  },
  rightText: {
    ...theme.typography.caption,
    color: theme.colors.textMuted
  }
});
