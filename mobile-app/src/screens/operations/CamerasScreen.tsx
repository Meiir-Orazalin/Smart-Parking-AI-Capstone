import React, { memo, useEffect, useMemo, useState } from "react";
import {
  Alert,
  ActivityIndicator,
  FlatList,
  Image,
  ListRenderItemInfo,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View
} from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { ScreenContainer } from "../../components/ScreenContainer";
import { AppCard } from "../../components/AppCard";
import { PrimaryButton } from "../../components/PrimaryButton";
import { StatusPill } from "../../components/StatusPill";
import { useDashboard } from "../../hooks/useDashboard";
import { useRoleSwitcher } from "../../hooks/useRoleSwitcher";
import { CameraItem } from "../../types/models";
import { colors, radii, spacing, typography } from "../../theme";

const LOW_FPS_THRESHOLD = 15;
const STALE_DETECTION_MINUTES = 10;
const STALE_DATA_THRESHOLD_MS = 45_000;
const FRESHNESS_TICK_MS = 10_000;
const RTSP_URL_PATTERN = /^rtsp:\/\/[^\s]+$/i;
const HTTP_STREAM_URL_PATTERN = /^https?:\/\/[^\s]+$/i;

type CameraFilter = "all" | "offline" | "low_fps" | "stale_detection";

type FilterChip = {
  id: CameraFilter;
  label: string;
  count: number;
};

type CameraFormErrors = {
  cameraName?: string;
  cameraLocation?: string;
  streamUrl?: string;
};

function parseDetectionAgeMinutes(lastDetection: string): number {
  const normalized = lastDetection.trim().toLowerCase();

  if (normalized === "just now") return 0;

  const secMatch = normalized.match(/(\d+)\s*sec/);
  if (secMatch) {
    return Number(secMatch[1]) / 60;
  }

  const minMatch = normalized.match(/(\d+)\s*min/);
  if (minMatch) {
    return Number(minMatch[1]);
  }

  const hourMatch = normalized.match(/(\d+)\s*hour/);
  if (hourMatch) {
    return Number(hourMatch[1]) * 60;
  }

  return Infinity;
}

function isLowFps(camera: CameraItem): boolean {
  return camera.fps > 0 && camera.fps < LOW_FPS_THRESHOLD;
}

function hasStaleDetection(camera: CameraItem): boolean {
  return parseDetectionAgeMinutes(camera.lastDetection) >= STALE_DETECTION_MINUTES;
}

function cameraHealthScore(camera: CameraItem): number {
  let score = 0;
  if (camera.status === "offline") score += 100;
  if (isLowFps(camera)) score += 50;
  if (hasStaleDetection(camera)) score += 30;
  return score;
}

function formatFreshness(lastSyncedAt: string | null, refreshing: boolean, clockMs: number): string {
  if (refreshing) return "Refreshing data...";
  if (!lastSyncedAt) return "Not synced yet";

  const ageSec = Math.max(0, Math.round((clockMs - new Date(lastSyncedAt).getTime()) / 1000));
  if (ageSec < 5) return "Updated just now";
  if (ageSec < 60) return `Updated ${ageSec}s ago`;
  return `Updated ${Math.round(ageSec / 60)}m ago`;
}

function isStaleData(lastSyncedAt: string | null, clockMs: number): boolean {
  if (!lastSyncedAt) return true;
  return clockMs - new Date(lastSyncedAt).getTime() > STALE_DATA_THRESHOLD_MS;
}

function healthTags(camera: CameraItem): string[] {
  const tags: string[] = [];
  if (camera.status === "offline") tags.push("Offline");
  if (isLowFps(camera)) tags.push("Low FPS");
  if (hasStaleDetection(camera)) tags.push("Stale Detection");
  return tags;
}

function validateCameraForm(
  cameraName: string,
  cameraLocation: string,
  streamUrl: string,
  cameras: CameraItem[]
): CameraFormErrors {
  const errors: CameraFormErrors = {};
  const trimmedName = cameraName.trim();
  const trimmedLocation = cameraLocation.trim();
  const trimmedRtsp = streamUrl.trim();

  if (!trimmedName) {
    errors.cameraName = "Camera name is required.";
  } else if (trimmedName.length < 3) {
    errors.cameraName = "Camera name must be at least 3 characters.";
  }

  if (!trimmedLocation) {
    errors.cameraLocation = "Location is required.";
  }

  if (!trimmedRtsp) {
    errors.streamUrl = "RTSP URL is required.";
  } else if (!RTSP_URL_PATTERN.test(trimmedRtsp)) {
    errors.streamUrl = "Use a valid RTSP URL (rtsp://...).";
  }

  const lowerName = trimmedName.toLowerCase();
  const lowerRtsp = trimmedRtsp.toLowerCase();

  if (lowerName && cameras.some((camera) => camera.name.trim().toLowerCase() === lowerName)) {
    errors.cameraName = "A camera with this name already exists.";
  }

  if (
    lowerRtsp &&
    cameras.some((camera) => (camera.streamUrl ?? "").trim().toLowerCase() === lowerRtsp)
  ) {
    errors.streamUrl = "This RTSP URL has already been added.";
  }

  return errors;
}

type CameraRowProps = {
  camera: CameraItem;
};

const CameraRow = memo(function CameraRow({ camera }: CameraRowProps) {
  const tags = healthTags(camera);
  const showStreamPreview =
    camera.status === "online" &&
    typeof camera.streamUrl === "string" &&
    HTTP_STREAM_URL_PATTERN.test(camera.streamUrl);

  return (
    <AppCard style={styles.rowCard}>
      <View style={styles.thumbnail}>
        {showStreamPreview ? (
          <Image
            source={{ uri: camera.streamUrl }}
            style={styles.thumbnailImage}
            resizeMode="cover"
            accessibilityIgnoresInvertColors
          />
        ) : (
          <MaterialCommunityIcons name="cctv" size={24} color={colors.textMuted} />
        )}
      </View>
      <View style={styles.info}>
        <View style={styles.headerRow}>
          <Text style={styles.name}>{camera.name}</Text>
          <StatusPill label={camera.status} />
        </View>
        <Text style={styles.meta}>
          FPS: {camera.fps} - Uptime: {camera.uptime}
        </Text>
        <Text style={styles.meta}>Last Detection: {camera.lastDetection}</Text>
        {tags.length > 0 ? (
          <View style={styles.tagsRow}>
            {tags.map((tag) => (
              <View key={`${camera.id}-${tag}`} style={styles.healthTag}>
                <Text style={styles.healthTagText}>{tag}</Text>
              </View>
            ))}
          </View>
        ) : null}
      </View>
    </AppCard>
  );
});

export function CamerasScreen() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [cameraName, setCameraName] = useState("");
  const [cameraLocation, setCameraLocation] = useState("");
  const [streamUrl, setStreamUrl] = useState("");
  const [formErrors, setFormErrors] = useState<CameraFormErrors>({});
  const [activeFilter, setActiveFilter] = useState<CameraFilter>("all");
  const [clockMs, setClockMs] = useState(() => Date.now());

  const { cameras, loading, refreshing, error, addCamera, refresh, lastSyncedAt } = useDashboard();
  const { switchRole } = useRoleSwitcher();

  useEffect(() => {
    const timer = setInterval(() => {
      setClockMs(Date.now());
    }, FRESHNESS_TICK_MS);

    return () => clearInterval(timer);
  }, []);

  const chips = useMemo<FilterChip[]>(() => {
    const offline = cameras.filter((camera) => camera.status === "offline").length;
    const lowFps = cameras.filter((camera) => isLowFps(camera)).length;
    const staleDetection = cameras.filter((camera) => hasStaleDetection(camera)).length;

    return [
      { id: "all", label: "All", count: cameras.length },
      { id: "offline", label: "Offline", count: offline },
      { id: "low_fps", label: "Low FPS", count: lowFps },
      { id: "stale_detection", label: "No Detection 10m+", count: staleDetection }
    ];
  }, [cameras]);

  const filteredCameras = useMemo(() => {
    const filtered = cameras.filter((camera) => {
      if (activeFilter === "offline") return camera.status === "offline";
      if (activeFilter === "low_fps") return isLowFps(camera);
      if (activeFilter === "stale_detection") return hasStaleDetection(camera);
      return true;
    });

    return [...filtered].sort((a, b) => cameraHealthScore(b) - cameraHealthScore(a));
  }, [cameras, activeFilter]);

  const freshnessLabel = formatFreshness(lastSyncedAt, refreshing, clockMs);
  const staleData = isStaleData(lastSyncedAt, clockMs);

  const closeModal = () => {
    setIsModalOpen(false);
    setCameraName("");
    setCameraLocation("");
    setStreamUrl("");
    setFormErrors({});
  };

  const saveCamera = async () => {
    const nextErrors = validateCameraForm(cameraName, cameraLocation, streamUrl, cameras);
    setFormErrors(nextErrors);

    if (Object.keys(nextErrors).length > 0) {
      return;
    }

    try {
      const trimmedName = cameraName.trim();
      await addCamera({
        name: trimmedName,
        location: cameraLocation.trim(),
        streamUrl: streamUrl.trim()
      });
      Alert.alert("Camera Added", `${trimmedName} is now visible in camera management.`);
      closeModal();
    } catch {
      Alert.alert("Camera Error", "Unable to add camera right now.");
    }
  };

  const renderCamera = ({ item }: ListRenderItemInfo<CameraItem>) => (
    <CameraRow camera={item} />
  );

  const header = (
    <View style={styles.header}>
      <Text style={styles.title}>Camera Management</Text>
      <Text style={styles.subtitle}>Monitor stream health and detection performance.</Text>

      <View style={styles.topActions}>
        <PrimaryButton label="Switch Role" variant="outline" onPress={switchRole} style={styles.flexButton} />
        <PrimaryButton label="Refresh" variant="outline" onPress={() => void refresh()} style={styles.flexButton} />
      </View>

      <View style={styles.syncRow}>
        <Text style={[styles.syncText, staleData && styles.staleText]}>{freshnessLabel}</Text>
        {staleData ? (
          <View style={styles.staleBadge}>
            <Text style={styles.staleBadgeText}>Stale</Text>
          </View>
        ) : null}
      </View>

      {loading || refreshing ? (
        <View style={styles.loadingRow}>
          <ActivityIndicator size="small" color={colors.accent} />
          <Text style={styles.loadingText}>{loading ? "Loading cameras..." : "Refreshing cameras..."}</Text>
        </View>
      ) : null}

      {error ? <Text style={styles.errorText}>{error}</Text> : null}

      <View style={styles.chipsWrap}>
        {chips.map((chip) => {
          const selected = activeFilter === chip.id;
          return (
            <Pressable
              key={chip.id}
              onPress={() => setActiveFilter(chip.id)}
              style={[styles.chip, selected && styles.chipSelected]}
              accessibilityRole="button"
              accessibilityLabel={`Filter cameras by ${chip.label}`}
            >
              <Text style={[styles.chipText, selected && styles.chipTextSelected]}>
                {chip.label} ({chip.count})
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );

  return (
    <>
      <ScreenContainer scroll={false}>
        <View style={styles.page}>
          <FlatList
            data={filteredCameras}
            renderItem={renderCamera}
            keyExtractor={(item) => item.id}
            ListHeaderComponent={header}
            ListEmptyComponent={
              <AppCard>
                <Text style={styles.emptyText}>No cameras match this filter.</Text>
              </AppCard>
            }
            ListFooterComponent={
              <View style={styles.footer}>
                <PrimaryButton label="Add new camera" onPress={() => setIsModalOpen(true)} />
              </View>
            }
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
            initialNumToRender={8}
            maxToRenderPerBatch={10}
            windowSize={8}
            removeClippedSubviews
            refreshing={refreshing}
            onRefresh={() => void refresh()}
          />
        </View>
      </ScreenContainer>

      <Modal visible={isModalOpen} transparent animationType="slide" onRequestClose={closeModal}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Add Camera</Text>

            <TextInput
              value={cameraName}
              onChangeText={(value) => {
                setCameraName(value);
                setFormErrors((current) => ({ ...current, cameraName: undefined }));
              }}
              placeholder="Camera name"
              placeholderTextColor={colors.textMuted}
              style={[styles.input, formErrors.cameraName && styles.inputError]}
              autoCapitalize="words"
            />
            {formErrors.cameraName ? <Text style={styles.fieldError}>{formErrors.cameraName}</Text> : null}

            <TextInput
              value={cameraLocation}
              onChangeText={(value) => {
                setCameraLocation(value);
                setFormErrors((current) => ({ ...current, cameraLocation: undefined }));
              }}
              placeholder="Location"
              placeholderTextColor={colors.textMuted}
              style={[styles.input, formErrors.cameraLocation && styles.inputError]}
              autoCapitalize="words"
            />
            {formErrors.cameraLocation ? <Text style={styles.fieldError}>{formErrors.cameraLocation}</Text> : null}

            <TextInput
              value={streamUrl}
              onChangeText={(value) => {
                setStreamUrl(value);
                setFormErrors((current) => ({ ...current, streamUrl: undefined }));
              }}
              placeholder="RTSP URL"
              placeholderTextColor={colors.textMuted}
              style={[styles.input, formErrors.streamUrl && styles.inputError]}
              autoCapitalize="none"
              autoCorrect={false}
            />
            {formErrors.streamUrl ? <Text style={styles.fieldError}>{formErrors.streamUrl}</Text> : null}

            <View style={styles.modalActions}>
              <PrimaryButton label="Cancel" variant="outline" onPress={closeModal} />
              <PrimaryButton label="Save" onPress={() => void saveCamera()} />
            </View>
          </View>
        </View>
      </Modal>
    </>
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
    gap: spacing.sm
  },
  header: {
    gap: spacing.md
  },
  footer: {
    marginTop: spacing.sm
  },
  title: {
    ...typography.h1,
    color: colors.textPrimary
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary
  },
  topActions: {
    flexDirection: "row",
    gap: spacing.sm
  },
  flexButton: {
    flex: 1
  },
  syncRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm
  },
  syncText: {
    ...typography.caption,
    color: colors.textMuted
  },
  staleText: {
    color: colors.warning
  },
  staleBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: radii.pill,
    backgroundColor: "rgba(247, 183, 49, 0.18)",
    borderWidth: 1,
    borderColor: "rgba(247, 183, 49, 0.35)"
  },
  staleBadgeText: {
    ...typography.caption,
    color: colors.warning,
    fontWeight: "700"
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
  chipsWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.xs
  },
  chip: {
    borderRadius: radii.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.md,
    paddingVertical: 8
  },
  chipSelected: {
    borderColor: colors.accent,
    backgroundColor: "rgba(77, 163, 255, 0.16)"
  },
  chipText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontWeight: "700"
  },
  chipTextSelected: {
    color: colors.textPrimary
  },
  rowCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md
  },
  thumbnail: {
    width: 66,
    height: 66,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surfaceAlt,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden"
  },
  thumbnailImage: {
    width: "100%",
    height: "100%"
  },
  info: {
    flex: 1,
    gap: 2
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.sm
  },
  name: {
    ...typography.bodyBold,
    color: colors.textPrimary,
    flex: 1
  },
  meta: {
    ...typography.caption,
    color: colors.textSecondary
  },
  tagsRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.xs,
    marginTop: spacing.xs
  },
  healthTag: {
    borderRadius: radii.pill,
    borderWidth: 1,
    borderColor: colors.warning,
    backgroundColor: "rgba(247, 183, 49, 0.16)",
    paddingHorizontal: spacing.sm,
    paddingVertical: 4
  },
  healthTagText: {
    ...typography.caption,
    color: colors.warning,
    fontWeight: "700"
  },
  emptyText: {
    ...typography.caption,
    color: colors.textMuted
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: colors.overlay,
    justifyContent: "flex-end"
  },
  modalCard: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: radii.lg,
    borderTopRightRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.sm
  },
  modalTitle: {
    ...typography.h3,
    color: colors.textPrimary,
    marginBottom: spacing.xs
  },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    backgroundColor: colors.surfaceAlt,
    color: colors.textPrimary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    ...typography.body
  },
  inputError: {
    borderColor: colors.danger
  },
  fieldError: {
    ...typography.caption,
    color: colors.danger
  },
  modalActions: {
    flexDirection: "row",
    gap: spacing.sm,
    marginTop: spacing.sm
  }
});
