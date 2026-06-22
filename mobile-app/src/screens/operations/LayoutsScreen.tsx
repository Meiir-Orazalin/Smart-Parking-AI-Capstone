import React, { useMemo, useRef, useState } from "react";
import { ActivityIndicator, Animated, Pressable, StyleSheet, Text, View } from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { ScreenContainer } from "../../components/ScreenContainer";
import { AppCard } from "../../components/AppCard";
import { PrimaryButton } from "../../components/PrimaryButton";
import { useLots } from "../../hooks/useLots";
import { useRoleSwitcher } from "../../hooks/useRoleSwitcher";
import { colors, radii, spacing, typography } from "../../theme";

const panelWidth = 280;

export function LayoutsScreen() {
  const { lots, loading, error } = useLots();
  const { switchRole } = useRoleSwitcher();
  const [selectedLotId, setSelectedLotId] = useState("main-lot");
  const [panelOpen, setPanelOpen] = useState(false);
  const slideX = useRef(new Animated.Value(-panelWidth)).current;

  const layoutLots = useMemo(
    () =>
      lots.filter((lot) =>
        ["RIT Dubai Dormitory", "Main Lot", "North Lot", "Visitor Garage - L2", "West Wing"].includes(lot.name)
      ),
    [lots]
  );

  const selectedLot = layoutLots.find((lot) => lot.id === selectedLotId) ?? layoutLots[0] ?? null;

  const openPanel = () => {
    setPanelOpen(true);
    Animated.timing(slideX, {
      toValue: 0,
      duration: 220,
      useNativeDriver: true
    }).start();
  };

  const closePanel = () => {
    Animated.timing(slideX, {
      toValue: -panelWidth,
      duration: 180,
      useNativeDriver: true
    }).start(({ finished }) => {
      if (finished) {
        setPanelOpen(false);
      }
    });
  };

  return (
    <View style={styles.page}>
      <ScreenContainer scroll={false}>
        <View style={styles.inner}>
          <Text style={styles.title}>Layout Management</Text>
          <Text style={styles.subtitle}>Choose a lot to inspect and annotate parking layout.</Text>
          <PrimaryButton label="Switch Role" variant="outline" onPress={switchRole} />

          {loading ? (
            <View style={styles.loadingRow}>
              <ActivityIndicator size="small" color={colors.accent} />
              <Text style={styles.loadingText}>Loading lot layouts...</Text>
            </View>
          ) : null}

          {error ? <Text style={styles.errorText}>{error}</Text> : null}

          <PrimaryButton label="Select Lot Panel" variant="outline" onPress={openPanel} />

          <AppCard style={styles.layoutCard}>
            <MaterialCommunityIcons name="map-search-outline" size={34} color={colors.textMuted} />
            <Text style={styles.layoutTitle}>{selectedLot?.name ?? "No Lot Selected"}</Text>
            <Text style={styles.layoutText}>Layout Preview Placeholder</Text>
          </AppCard>
        </View>
      </ScreenContainer>

      {panelOpen ? <Pressable style={styles.overlay} onPress={closePanel} /> : null}

      <Animated.View style={[styles.panel, { transform: [{ translateX: slideX }] }]}>
        <Text style={styles.panelTitle}>Lots</Text>
        <View style={styles.panelList}>
          {layoutLots.map((lot) => {
            const selected = lot.id === selectedLot?.id;
            return (
              <Pressable
                key={lot.id}
                onPress={() => {
                  setSelectedLotId(lot.id);
                  closePanel();
                }}
                style={[styles.panelItem, selected && styles.panelItemSelected]}
              >
                <Text style={[styles.panelItemText, selected && styles.panelItemTextSelected]}>{lot.name}</Text>
              </Pressable>
            );
          })}
        </View>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  page: {
    flex: 1,
    backgroundColor: colors.background
  },
  inner: {
    flex: 1,
    padding: spacing.lg,
    gap: spacing.md
  },
  title: {
    ...typography.h1,
    color: colors.textPrimary
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary
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
  errorText: {
    ...typography.caption,
    color: colors.danger
  },
  layoutCard: {
    flex: 1,
    minHeight: 300,
    borderRadius: radii.lg,
    backgroundColor: colors.surfaceAlt,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm
  },
  layoutTitle: {
    ...typography.h3,
    color: colors.textPrimary
  },
  layoutText: {
    ...typography.caption,
    color: colors.textMuted
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: colors.overlay
  },
  panel: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    width: panelWidth,
    backgroundColor: colors.surface,
    borderRightColor: colors.border,
    borderRightWidth: 1,
    paddingTop: 56,
    paddingHorizontal: spacing.lg
  },
  panelTitle: {
    ...typography.h3,
    color: colors.textPrimary,
    marginBottom: spacing.md
  },
  panelList: {
    gap: spacing.sm
  },
  panelItem: {
    backgroundColor: colors.surfaceAlt,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm
  },
  panelItemSelected: {
    borderColor: colors.accent
  },
  panelItemText: {
    ...typography.body,
    color: colors.textSecondary
  },
  panelItemTextSelected: {
    color: colors.textPrimary
  }
});
