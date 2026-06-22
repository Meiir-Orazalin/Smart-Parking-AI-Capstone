import React, { useEffect, useState } from "react";
import { ActivityIndicator, Alert, StyleSheet, Switch, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { ScreenContainer } from "../../components/ScreenContainer";
import { AppCard } from "../../components/AppCard";
import { ListRow } from "../../components/ListRow";
import { EnvironmentBanner } from "../../components/EnvironmentBanner";
import { useDashboard } from "../../hooks/useDashboard";
import { useRoleSwitcher } from "../../hooks/useRoleSwitcher";
import { colors, spacing, typography } from "../../theme";

export function ProfileScreen() {
  const { profile, loading, localOnlyData, resetLocalDemoData } = useDashboard();
  const { switchRole } = useRoleSwitcher();
  const [isResetting, setIsResetting] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(
    profile?.notificationsEnabled ?? true
  );

  useEffect(() => {
    if (profile) {
      setNotificationsEnabled(profile.notificationsEnabled);
    }
  }, [profile]);

  const handleResetDemoData = () => {
    Alert.alert(
      "Reset Demo Data",
      "This clears locally saved cameras and alert workflow on this device. Continue?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset",
          style: "destructive",
          onPress: async () => {
            try {
              setIsResetting(true);
              await resetLocalDemoData();
              Alert.alert("Reset Complete", "Local demo data has been cleared.");
            } finally {
              setIsResetting(false);
            }
          }
        }
      ]
    );
  };

  return (
    <ScreenContainer>
      <Text style={styles.title}>Profile</Text>
      <EnvironmentBanner />

      {loading ? (
        <View style={styles.loadingRow}>
          <ActivityIndicator size="small" color={colors.accent} />
          <Text style={styles.loadingText}>Loading profile...</Text>
        </View>
      ) : null}

      {isResetting ? (
        <View style={styles.loadingRow}>
          <ActivityIndicator size="small" color={colors.warning} />
          <Text style={styles.loadingText}>Resetting local demo data...</Text>
        </View>
      ) : null}

      <AppCard style={styles.profileCard}>
        <View style={styles.avatar}>
          <Ionicons name="person" size={28} color={colors.textPrimary} />
        </View>
        <View>
          <Text style={styles.name}>{profile?.name ?? "Meiir Orazalin"}</Text>
          <Text style={styles.email}>{profile?.email ?? "meiirorazalin@gmail.com"}</Text>
        </View>
      </AppCard>

      <AppCard style={styles.listCard}>
        <ListRow
          title="Parking History"
          subtitle="Recent spots and dwell time"
          onPress={() => Alert.alert("Parking History", "Prototype list only.")}
        />
        <ListRow
          title="Notifications"
          subtitle="Lot updates and recommendations"
          rightNode={
            <Switch
              value={notificationsEnabled}
              onValueChange={setNotificationsEnabled}
              trackColor={{ true: "rgba(77, 163, 255, 0.55)", false: colors.surfaceAlt }}
              thumbColor={notificationsEnabled ? colors.accent : colors.textMuted}
            />
          }
        />
        <ListRow
          title="Switch Role"
          subtitle="Return to mode selection"
          onPress={switchRole}
        />
        {localOnlyData ? (
          <ListRow
            title="Reset Demo Data"
            subtitle="Clear local cameras and alert workflow"
            onPress={handleResetDemoData}
          />
        ) : null}
        <ListRow title="Log Out" subtitle="Exit current session" onPress={() => Alert.alert("Logged Out")} />
      </AppCard>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  title: {
    ...typography.h1,
    color: colors.textPrimary
  },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm
  },
  loadingText: {
    ...typography.caption,
    color: colors.textMuted
  },
  profileCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md
  },
  avatar: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: colors.surfaceAlt,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: "center",
    justifyContent: "center"
  },
  name: {
    ...typography.h3,
    color: colors.textPrimary
  },
  email: {
    ...typography.caption,
    color: colors.textSecondary
  },
  listCard: {
    paddingTop: spacing.sm
  }
});
