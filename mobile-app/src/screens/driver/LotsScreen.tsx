import React, { memo, useCallback, useMemo, useState } from "react";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import {
  ActivityIndicator,
  FlatList,
  ListRenderItemInfo,
  Pressable,
  StyleSheet,
  Text,
  View
} from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { DriverLotsStackParamList } from "../../navigation/types";
import { ScreenContainer } from "../../components/ScreenContainer";
import { SearchBar } from "../../components/SearchBar";
import { AppCard } from "../../components/AppCard";
import { ProgressBar } from "../../components/ProgressBar";
import { StatusPill } from "../../components/StatusPill";
import { useLots } from "../../hooks/useLots";
import { ParkingLot } from "../../types/models";
import { colors, spacing, typography } from "../../theme";

type Props = NativeStackScreenProps<DriverLotsStackParamList, "Lots">;

function formatUpdatedLabel(iso: string) {
  const timestamp = new Date(iso).getTime();
  const diffMinutes = Math.max(0, Math.round((Date.now() - timestamp) / 60000));
  return `${diffMinutes} min ago`;
}

type LotRowProps = {
  lot: ParkingLot;
  onPress: (lotId: string) => void;
};

const LotRow = memo(function LotRow({ lot, onPress }: LotRowProps) {
  return (
    <Pressable
      onPress={() => onPress(lot.id)}
      accessibilityRole="button"
      accessibilityLabel={`Open details for ${lot.name}`}
    >
      <AppCard style={styles.lotCard}>
        <View style={styles.rowBetween}>
          <Text style={styles.lotName}>{lot.name}</Text>
          <StatusPill label={lot.status} />
        </View>

        <Text style={styles.occupancy}>
          {lot.occupied}/{lot.capacity} Occupied
        </Text>
        <ProgressBar value={lot.occupied} max={lot.capacity} status={lot.status} />

        <View style={styles.metaRow}>
          <View style={styles.metaCell}>
            <MaterialCommunityIcons name="map-marker-distance" size={15} color={colors.textMuted} />
            <Text style={styles.metaText}>{lot.distanceMi} mi</Text>
          </View>
          <View style={styles.metaCell}>
            <MaterialCommunityIcons name="clock-outline" size={15} color={colors.textMuted} />
            <Text style={styles.metaText}>{formatUpdatedLabel(lot.lastUpdated)}</Text>
          </View>
        </View>
      </AppCard>
    </Pressable>
  );
});

export function LotsScreen({ navigation }: Props) {
  const [query, setQuery] = useState("");
  const { lots, loading, refreshing, error, refresh } = useLots();

  const filteredLots = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return lots.filter((lot) =>
        ["RIT Dubai Dormitory", "Main Lot", "North Lot", "Central Garage", "West Plaza", "South Deck"].includes(lot.name)
      );
    }
    return lots.filter((lot) => lot.name.toLowerCase().includes(normalized));
  }, [lots, query]);

  const handleLotPress = useCallback(
    (lotId: string) => {
      navigation.navigate("LotDetails", { lotId });
    },
    [navigation]
  );

  const renderLot = useCallback(
    ({ item }: ListRenderItemInfo<ParkingLot>) => (
      <LotRow lot={item} onPress={handleLotPress} />
    ),
    [handleLotPress]
  );

  const header = (
    <View style={styles.header}>
      <Text style={styles.title}>Select Parking Lot</Text>
      <Text style={styles.subtitle}>Real-time occupancy powered by SmartPark AI</Text>
      <SearchBar value={query} onChangeText={setQuery} placeholder="Search parking lots..." />

      {loading ? (
        <View style={styles.loadingRow}>
          <ActivityIndicator size="small" color={colors.accent} />
          <Text style={styles.loadingText}>Loading lot data...</Text>
        </View>
      ) : null}

      {error ? <Text style={styles.errorText}>{error}</Text> : null}
    </View>
  );

  return (
    <ScreenContainer scroll={false}>
      <View style={styles.page}>
        <FlatList
          data={filteredLots}
          renderItem={renderLot}
          keyExtractor={(item) => item.id}
          ListHeaderComponent={header}
          ListEmptyComponent={
            <AppCard>
              <Text style={styles.emptyText}>No parking lots match your search.</Text>
            </AppCard>
          }
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          initialNumToRender={6}
          maxToRenderPerBatch={8}
          windowSize={8}
          removeClippedSubviews
          refreshing={refreshing}
          onRefresh={() => void refresh()}
        />
      </View>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  page: {
    flex: 1
  },
  listContent: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xl,
    gap: spacing.md
  },
  header: {
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
  emptyText: {
    ...typography.caption,
    color: colors.textMuted
  },
  lotCard: {
    gap: spacing.sm
  },
  rowBetween: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center"
  },
  lotName: {
    ...typography.h3,
    color: colors.textPrimary,
    flex: 1,
    marginRight: spacing.sm
  },
  occupancy: {
    ...typography.bodyBold,
    color: colors.textPrimary
  },
  metaRow: {
    flexDirection: "row",
    gap: spacing.md,
    marginTop: spacing.xs
  },
  metaCell: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs
  },
  metaText: {
    ...typography.caption,
    color: colors.textMuted
  }
});
