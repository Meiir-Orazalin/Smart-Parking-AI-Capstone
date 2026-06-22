import React, { useEffect, useMemo, useRef, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppCard } from "./AppCard";
import {
  BusynessDay,
  BusynessPoint,
  getBusynessPrediction
} from "../services/busynessPredictions";
import { ParkingLot } from "../types/models";
import { colors, radii, spacing, typography } from "../theme";

type BusynessChartCardProps = {
  lot: ParkingLot;
};

const dayOptions: { label: string; value: BusynessDay }[] = [
  { label: "Today", value: "today" },
  { label: "Tomorrow", value: "tomorrow" },
  { label: "Weekend", value: "weekend" }
];

const barSlotWidth = 34;

function formatHour(hour: number): string {
  if (hour === 0) return "12 AM";
  if (hour < 12) return `${hour} AM`;
  if (hour === 12) return "12 PM";
  return `${hour - 12} PM`;
}

function levelForPct(value: number): { label: string; color: string } {
  if (value >= 85) return { label: "Very Busy", color: colors.danger };
  if (value >= 70) return { label: "Busy", color: colors.warning };
  if (value >= 40) return { label: "Moderate", color: colors.accent };
  return { label: "Low", color: colors.success };
}

function summarize(points: BusynessPoint[]): string {
  const peak = points.reduce((highest, point) => (point.busynessPct > highest.busynessPct ? point : highest), points[0]);
  return `Peak demand is expected around ${formatHour(peak.hour)}.`;
}

function guidanceForLot(lot: ParkingLot, currentOccupancyPct: number): { text: string; color: string } {
  if (lot.status === "full" || currentOccupancyPct >= 95) {
    return {
      text: "This lot is at capacity right now. Check another lot before driving in.",
      color: colors.danger
    };
  }

  if (lot.status === "almost_full" || currentOccupancyPct >= 80) {
    return {
      text: "Demand is high. Arrive soon or compare nearby lots.",
      color: colors.warning
    };
  }

  return {
    text: "Spaces are likely available. Demand is manageable right now.",
    color: colors.success
  };
}

export function BusynessChartCard({ lot }: BusynessChartCardProps) {
  const [selectedDay, setSelectedDay] = useState<BusynessDay>("today");
  const [dayMenuOpen, setDayMenuOpen] = useState(false);
  const scrollRef = useRef<ScrollView>(null);

  const points = useMemo(() => getBusynessPrediction(lot, selectedDay), [lot, selectedDay]);

  const initialIndex = useMemo(() => {
    const currentHour = new Date().getHours();
    const exactIndex = points.findIndex((point) => point.hour >= currentHour);
    if (exactIndex >= 0) return exactIndex;
    return Math.max(0, points.length - 1);
  }, [points]);
  const [selectedIndex, setSelectedIndex] = useState(initialIndex);

  useEffect(() => {
    setSelectedIndex(initialIndex);
  }, [initialIndex, lot.id, selectedDay]);

  useEffect(() => {
    const scrollX = Math.max(0, (initialIndex - 3) * barSlotWidth);
    const timer = setTimeout(() => {
      scrollRef.current?.scrollTo({ x: scrollX, animated: true });
    }, 80);

    return () => clearTimeout(timer);
  }, [initialIndex, lot.id, selectedDay]);

  const selected = points[selectedIndex] ?? points[0];
  const level = levelForPct(selected.busynessPct);
  const expectedFree = Math.max(0, Math.round(lot.capacity * (1 - selected.busynessPct / 100)));
  const currentOccupancyPct = lot.capacity === 0 ? 0 : Math.round((lot.occupied / lot.capacity) * 100);
  const guidance = guidanceForLot(lot, currentOccupancyPct);
  const averageStayHours = selected.busynessPct >= 70 ? "1.5-2.5 h" : selected.busynessPct >= 40 ? "45 min-1.5 h" : "20-45 min";
  const selectedDayLabel = dayOptions.find((option) => option.value === selectedDay)?.label ?? "Today";
  const currentHour = new Date().getHours();

  return (
    <AppCard style={styles.card}>
      <View style={[styles.header, dayMenuOpen && styles.headerRaised]}>
        <View style={styles.headerTitle}>
          <Ionicons name="people-outline" size={20} color={colors.textSecondary} />
          <Text style={styles.title}>Busyness Level</Text>
        </View>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`Change busyness day, selected ${selectedDayLabel}`}
          onPress={() => setDayMenuOpen((open) => !open)}
          style={styles.dayBadge}
        >
          <Text style={styles.dayBadgeText}>{selectedDayLabel.toUpperCase()}</Text>
          <Ionicons name={dayMenuOpen ? "chevron-up" : "chevron-down"} size={14} color={colors.textPrimary} />
        </Pressable>

        {dayMenuOpen ? (
          <View style={styles.dayMenu}>
            {dayOptions.map((option) => {
              const selectedOption = option.value === selectedDay;
              return (
                <Pressable
                  key={option.value}
                  onPress={() => {
                    setSelectedDay(option.value);
                    setDayMenuOpen(false);
                  }}
                  style={[styles.dayMenuItem, selectedOption && styles.dayMenuItemSelected]}
                >
                  <Text style={[styles.dayMenuText, selectedOption && styles.dayMenuTextSelected]}>
                    {option.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        ) : null}
      </View>

      <Text style={styles.helper}>Current time is selected automatically. Swipe for other hours.</Text>

      <ScrollView
        ref={scrollRef}
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.chart}
      >
          {points.map((point, index) => {
            const selectedBar = index === selectedIndex;
            const height = 18 + point.busynessPct * 0.52;
            const showTick = [0, 3, 6, 9, 12, 15, 18, 21].includes(point.hour);
            const barLevel = levelForPct(point.busynessPct);
            const isCurrentHour = selectedDay === "today" && point.hour === currentHour;

            return (
              <View key={`${selectedDay}-${point.hour}`} style={styles.barSlot}>
                <Text style={[styles.nowLabel, isCurrentHour && styles.nowLabelVisible]}>
                  {isCurrentHour ? "Now" : " "}
                </Text>
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel={`${formatHour(point.hour)} predicted busyness ${point.busynessPct} percent, ${barLevel.label}`}
                  onPress={() => setSelectedIndex(index)}
                  style={styles.barPressArea}
                >
                  <View
                    style={[
                      styles.bar,
                      {
                        height,
                        backgroundColor: selectedBar ? barLevel.color : "rgba(123, 160, 174, 0.9)"
                      },
                      selectedBar && styles.selectedBar
                    ]}
                  />
                </Pressable>
                <Text style={[styles.tick, showTick && styles.tickVisible]}>{showTick ? point.hour : " "}</Text>
              </View>
            );
          })}
      </ScrollView>

      <View style={styles.dots}>
        {[0, 1, 2, 3, 4, 5, 6].map((dot) => (
          <View key={dot} style={[styles.dot, dot === 4 && styles.dotActive]} />
        ))}
      </View>

      <Text style={styles.summary}>{summarize(points)}</Text>

      <View style={[styles.guidanceBox, { borderColor: guidance.color }]}>
        <View style={styles.guidanceTopRow}>
          <Text style={styles.guidanceLabel}>Current occupancy</Text>
          <Text style={[styles.guidanceValue, { color: guidance.color }]}>{currentOccupancyPct}%</Text>
        </View>
        <Text style={styles.guidanceText}>{guidance.text}</Text>
      </View>

      <View style={styles.detailRow}>
        <View>
          <Text style={styles.detailLabel}>{formatHour(selected.hour)}</Text>
          <Text style={[styles.levelText, { color: level.color }]}>{level.label}</Text>
        </View>
        <View style={styles.detailRight}>
          <Text style={styles.detailValue}>{selected.busynessPct}% busy</Text>
          <Text style={styles.detailHint}>{expectedFree} spaces likely free</Text>
        </View>
      </View>

      <Text style={styles.stayText}>Typical stay: {averageStayHours}</Text>
    </AppCard>
  );
}

const styles = StyleSheet.create({
  card: {
    gap: spacing.md,
    overflow: "visible"
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.md,
    position: "relative",
    zIndex: 2
  },
  headerRaised: {
    marginBottom: 90
  },
  headerTitle: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    flex: 1
  },
  title: {
    ...typography.bodyBold,
    color: colors.textPrimary
  },
  dayBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    borderRadius: radii.md,
    backgroundColor: "rgba(230, 237, 247, 0.16)",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm
  },
  dayBadgeText: {
    ...typography.caption,
    color: colors.textPrimary,
    fontWeight: "700"
  },
  dayMenu: {
    position: "absolute",
    top: 42,
    right: 0,
    width: 142,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: spacing.xs,
    zIndex: 3
  },
  dayMenuItem: {
    borderRadius: radii.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm
  },
  dayMenuItemSelected: {
    backgroundColor: "rgba(77, 163, 255, 0.16)"
  },
  dayMenuText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontWeight: "700"
  },
  dayMenuTextSelected: {
    color: colors.textPrimary
  },
  helper: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: spacing.xs
  },
  chart: {
    minHeight: 132,
    flexDirection: "row",
    alignItems: "flex-end",
    paddingTop: spacing.md,
    paddingHorizontal: spacing.xs
  },
  barSlot: {
    alignItems: "center",
    width: barSlotWidth
  },
  barPressArea: {
    height: 88,
    justifyContent: "flex-end",
    alignItems: "center",
    width: "100%"
  },
  nowLabel: {
    ...typography.caption,
    color: "transparent",
    fontWeight: "700",
    minHeight: 18,
    marginBottom: spacing.xs
  },
  nowLabelVisible: {
    color: colors.accent
  },
  bar: {
    width: 18,
    borderRadius: radii.pill
  },
  selectedBar: {
    width: 20,
    shadowColor: colors.accent,
    shadowOpacity: 0.28,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 0 }
  },
  tick: {
    ...typography.caption,
    color: "transparent",
    marginTop: spacing.xs,
    minHeight: 18
  },
  tickVisible: {
    color: colors.textSecondary
  },
  dots: {
    flexDirection: "row",
    justifyContent: "center",
    gap: spacing.sm
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: "rgba(159, 176, 198, 0.45)"
  },
  dotActive: {
    backgroundColor: colors.textPrimary
  },
  summary: {
    ...typography.bodyBold,
    color: colors.textPrimary,
    textAlign: "center"
  },
  guidanceBox: {
    borderRadius: radii.md,
    borderWidth: 1,
    backgroundColor: "rgba(24, 35, 58, 0.72)",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    gap: spacing.xs
  },
  guidanceTopRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: spacing.md
  },
  guidanceLabel: {
    ...typography.caption,
    color: colors.textSecondary
  },
  guidanceValue: {
    ...typography.bodyBold
  },
  guidanceText: {
    ...typography.caption,
    color: colors.textPrimary
  },
  detailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingTop: spacing.md
  },
  detailLabel: {
    ...typography.caption,
    color: colors.textSecondary
  },
  levelText: {
    ...typography.h3
  },
  detailRight: {
    alignItems: "flex-end",
    flexShrink: 1
  },
  detailValue: {
    ...typography.bodyBold,
    color: colors.textPrimary
  },
  detailHint: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: "right"
  },
  stayText: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: "center"
  }
});
