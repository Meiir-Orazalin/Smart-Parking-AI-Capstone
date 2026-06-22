import React from "react";
import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { theme } from "../constants/theme";

type ListRowProps = {
  title: string;
  subtitle?: string;
  rightNode?: React.ReactNode;
  onPress?: () => void;
};

export function ListRow({ title, subtitle, rightNode, onPress }: ListRowProps) {
  return (
    <Pressable onPress={onPress} style={styles.row}>
      <View style={styles.left}>
        <Text style={styles.title}>{title}</Text>
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>

      {rightNode ?? <Ionicons name="chevron-forward" size={16} color={theme.colors.textMuted} />}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: theme.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border
  },
  left: {
    flex: 1,
    paddingRight: theme.spacing.md
  },
  title: {
    ...theme.typography.bodyBold,
    color: theme.colors.textPrimary
  },
  subtitle: {
    ...theme.typography.caption,
    color: theme.colors.textSecondary,
    marginTop: 2
  }
});
