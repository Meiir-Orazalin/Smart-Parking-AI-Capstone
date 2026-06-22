import React from "react";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { StyleSheet, Text, View } from "react-native";
import { RootStackParamList } from "../navigation/types";
import { ScreenContainer } from "../components/ScreenContainer";
import { PrimaryButton } from "../components/PrimaryButton";
import { AppCard } from "../components/AppCard";
import { EnvironmentBanner } from "../components/EnvironmentBanner";
import { colors, spacing, typography } from "../theme";

type Props = NativeStackScreenProps<RootStackParamList, "ChooseMode">;

export function ChooseModeScreen({ navigation }: Props) {
  return (
    <ScreenContainer contentStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.kicker}>SmartPark AI</Text>
        <Text style={styles.title}>Choose Mode</Text>
        <Text style={styles.subtitle}>Select your role to continue</Text>
      </View>

      <EnvironmentBanner />

      <AppCard style={styles.card}>
        <Text style={styles.cardTitle}>Driver Mode</Text>
        <Text style={styles.cardText}>Find open spaces faster with live lot occupancy and guidance.</Text>
        <PrimaryButton label="Enter Driver Mode" onPress={() => navigation.replace("DriverMode")} />
      </AppCard>

      <AppCard style={styles.card}>
        <Text style={styles.cardTitle}>Operations Mode</Text>
        <Text style={styles.cardText}>Monitor cameras, lot health, occupancy trends, and active alerts.</Text>
        <PrimaryButton
          label="Enter Operations Mode"
          variant="outline"
          onPress={() => navigation.replace("OperationsMode")}
        />
      </AppCard>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  content: {
    flexGrow: 1,
    justifyContent: "center",
    gap: spacing.lg
  },
  header: {
    marginBottom: spacing.sm
  },
  kicker: {
    ...typography.caption,
    color: colors.accent,
    letterSpacing: 1.5,
    textTransform: "uppercase"
  },
  title: {
    ...typography.h1,
    color: colors.textPrimary,
    marginTop: spacing.xs
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary,
    marginTop: spacing.xs
  },
  card: {
    gap: spacing.md
  },
  cardTitle: {
    ...typography.h3,
    color: colors.textPrimary
  },
  cardText: {
    ...typography.body,
    color: colors.textSecondary
  }
});
