import React from "react";
import { Pressable, StyleSheet, Text, ViewStyle } from "react-native";
import { theme } from "../constants/theme";

type PrimaryButtonProps = {
  label: string;
  onPress: () => void;
  style?: ViewStyle;
  variant?: "solid" | "outline";
};

export function PrimaryButton({
  label,
  onPress,
  style,
  variant = "solid"
}: PrimaryButtonProps) {
  return (
    <Pressable
      onPress={onPress}
      style={[styles.button, variant === "outline" ? styles.outline : styles.solid, style]}
    >
      <Text style={[styles.label, variant === "outline" ? styles.outlineLabel : styles.solidLabel]}>
        {label}
      </Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    borderRadius: theme.radii.md,
    minHeight: 52,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: theme.spacing.lg
  },
  solid: {
    backgroundColor: theme.colors.accent
  },
  outline: {
    borderWidth: 1,
    borderColor: theme.colors.border,
    backgroundColor: theme.colors.card
  },
  label: {
    ...theme.typography.bodyBold
  },
  solidLabel: {
    color: theme.colors.white
  },
  outlineLabel: {
    color: theme.colors.textPrimary
  }
});
