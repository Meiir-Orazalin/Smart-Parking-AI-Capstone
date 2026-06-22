from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .cache import SlotCache, SlotView
from .matching import ORBViewMatcher, ViewMatchResult, load_reference_frame


class FrameStabilityTracker:
    def __init__(
        self,
        *,
        min_stable_frames: int = 12,
        motion_threshold: float = 6.0,
        sample_size: tuple[int, int] = (320, 180),
    ) -> None:
        self.min_stable_frames = max(2, int(min_stable_frames))
        self.motion_threshold = float(motion_threshold)
        self.sample_size = sample_size
        self._previous_gray: np.ndarray | None = None
        self._stable_count = 0
        self._frames: deque[np.ndarray] = deque(maxlen=self.min_stable_frames)

    def reset(self) -> None:
        self._previous_gray = None
        self._stable_count = 0
        self._frames.clear()

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, self.sample_size)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def update(self, frame: np.ndarray) -> bool:
        gray = self._prepare_gray(frame)
        if self._previous_gray is None:
            self._previous_gray = gray
            self._stable_count = 1
            self._frames.clear()
            self._frames.append(frame.copy())
            return False

        diff = cv2.absdiff(gray, self._previous_gray)
        motion_score = float(diff.mean())
        self._previous_gray = gray
        if motion_score <= self.motion_threshold:
            self._stable_count += 1
            self._frames.append(frame.copy())
        else:
            self._stable_count = 0
            self._frames.clear()
        return self.is_stable

    @property
    def is_stable(self) -> bool:
        return self._stable_count >= self.min_stable_frames and len(self._frames) >= self.min_stable_frames

    def frames(self) -> list[np.ndarray]:
        return list(self._frames)

    def reference_frame(self) -> np.ndarray | None:
        if not self._frames:
            return None
        return self._frames[-1]


@dataclass(frozen=True)
class ViewStateSnapshot:
    state: str
    active_view_id: str | None
    best_view_id: str | None
    best_score: float
    motion_score: float
    stable_frames: int
    unstable_frames: int
    should_calibrate: bool
    should_fallback: bool
    match_result: ViewMatchResult | None = None
    note: str | None = None


class ViewStateManager:
    def __init__(
        self,
        cache: SlotCache,
        *,
        matcher: ORBViewMatcher | None = None,
        stable_motion_threshold: float = 2.5,
        stable_frames_required: int = 3,
        lost_frames_required: int = 3,
        switch_motion_threshold: float = 8.0,
        match_margin: float = 0.05,
        unsupported_motion_limit: int = 20,
    ) -> None:
        self.cache = cache
        self.matcher = matcher or ORBViewMatcher()
        self.stable_motion_threshold = float(stable_motion_threshold)
        self.stable_frames_required = int(stable_frames_required)
        self.lost_frames_required = int(lost_frames_required)
        self.switch_motion_threshold = float(switch_motion_threshold)
        self.match_margin = float(match_margin)
        self.unsupported_motion_limit = int(unsupported_motion_limit)

        self.active_view_id: str | None = None
        self.state = "warming_up"
        self.stable_frames = 0
        self.unstable_frames = 0
        self.lost_match_frames = 0
        self.frame_index = 0
        self._previous_motion_frame: np.ndarray | None = None
        self._last_match: ViewMatchResult | None = None

    def set_cache(self, cache: SlotCache) -> None:
        self.cache = cache
        self.clear()

    def clear(self) -> None:
        self.active_view_id = None
        self.state = "warming_up"
        self.stable_frames = 0
        self.unstable_frames = 0
        self.lost_match_frames = 0
        self.frame_index = 0
        self._previous_motion_frame = None
        self._last_match = None

    def set_active_view(self, view_id: str | None) -> None:
        self.active_view_id = view_id
        self.state = "ready" if view_id is not None else "warming_up"
        self.lost_match_frames = 0

    @property
    def last_match(self) -> ViewMatchResult | None:
        return self._last_match

    @property
    def cache_ready(self) -> bool:
        return bool(self.cache.views)

    def _prepare_motion_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        height, width = gray.shape[:2]
        max_dim = max(height, width)
        if max_dim > 320:
            scale = 320 / float(max_dim)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray

    def _motion_score(self, frame: np.ndarray) -> float:
        current = self._prepare_motion_frame(frame)
        if self._previous_motion_frame is None:
            self._previous_motion_frame = current
            return 0.0
        if self._previous_motion_frame.shape != current.shape:
            previous = cv2.resize(self._previous_motion_frame, (current.shape[1], current.shape[0]))
        else:
            previous = self._previous_motion_frame
        diff = cv2.absdiff(current, previous)
        score = float(np.mean(diff))
        self._previous_motion_frame = current
        return score

    def _load_reference(self, view: SlotView) -> np.ndarray | None:
        return load_reference_frame(self.cache.resolve_reference_frame_path(view) or view.reference_frame_path)

    def _best_match(self, frame: np.ndarray) -> ViewMatchResult:
        return self.matcher.match_cache(frame, self.cache, reference_loader=self._load_reference)

    def observe(self, frame: np.ndarray) -> ViewStateSnapshot:
        self.frame_index += 1
        motion_score = self._motion_score(frame)
        stable = motion_score <= self.stable_motion_threshold

        if stable:
            self.stable_frames += 1
            self.unstable_frames = 0
        else:
            self.unstable_frames += 1
            self.stable_frames = 0

        should_calibrate = False
        should_fallback = False
        best_match: ViewMatchResult | None = None

        if self.unstable_frames >= self.unsupported_motion_limit:
            self.state = "unsupported_motion"
            should_fallback = True
            self._last_match = None
            return ViewStateSnapshot(
                state=self.state,
                active_view_id=self.active_view_id,
                best_view_id=None,
                best_score=0.0,
                motion_score=motion_score,
                stable_frames=self.stable_frames,
                unstable_frames=self.unstable_frames,
                should_calibrate=False,
                should_fallback=should_fallback,
                match_result=None,
                note="motion_exceeded_limit",
            )

        if self.stable_frames >= self.stable_frames_required and self.cache.views:
            best_match = self._best_match(frame)
            self._last_match = best_match

        if self.active_view_id is not None:
            active_view = self.cache.get_view(self.active_view_id)
            active_reference = self._load_reference(active_view) if active_view is not None else None
            active_match = None
            if active_reference is not None and self.stable_frames >= self.stable_frames_required:
                active_match = self.matcher.score_pair(frame, active_reference)
                active_match = ViewMatchResult(
                    view_id=self.active_view_id,
                    score=active_match.score,
                    good_matches=active_match.good_matches,
                    keypoints_frame=active_match.keypoints_frame,
                    keypoints_reference=active_match.keypoints_reference,
                    matched=active_match.matched,
                    reference_frame_path=active_view.reference_frame_path if active_view else None,
                    reason=active_match.reason,
                )

            if active_match is not None and active_match.score >= self.matcher.min_score:
                self.lost_match_frames = 0
                if best_match is not None and best_match.view_id and best_match.view_id != self.active_view_id:
                    if best_match.score >= active_match.score + self.match_margin:
                        self.active_view_id = best_match.view_id
                        self.state = "ready"
                        self._last_match = best_match
                    else:
                        self.state = "ready"
                else:
                    self.state = "ready"
            else:
                self.lost_match_frames += 1
                if self.lost_match_frames >= self.lost_frames_required:
                    self.active_view_id = None
                    self.state = "view_switching" if stable else "warming_up"

        if self.active_view_id is None:
            if best_match is not None and best_match.matched:
                self.active_view_id = best_match.view_id
                self.state = "ready"
                self.lost_match_frames = 0
            elif stable and self.stable_frames >= self.stable_frames_required:
                self.state = "calibration_pending"
                should_calibrate = True
            elif not stable:
                self.state = "view_switching" if self.unstable_frames else "warming_up"
            else:
                self.state = "warming_up"

        if self.state == "view_switching" and best_match is not None and best_match.matched:
            self.active_view_id = best_match.view_id
            self.state = "ready"
            self.lost_match_frames = 0

        if self.active_view_id is None and self.state == "calibration_pending":
            should_fallback = not should_calibrate

        return ViewStateSnapshot(
            state=self.state,
            active_view_id=self.active_view_id,
            best_view_id=best_match.view_id if best_match is not None else None,
            best_score=best_match.score if best_match is not None else 0.0,
            motion_score=motion_score,
            stable_frames=self.stable_frames,
            unstable_frames=self.unstable_frames,
            should_calibrate=should_calibrate,
            should_fallback=should_fallback,
            match_result=best_match,
            note=None,
        )
