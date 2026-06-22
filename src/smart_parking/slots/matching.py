from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .cache import SlotCache, SlotView


@dataclass(frozen=True)
class ViewMatchResult:
    view_id: str | None
    score: float
    good_matches: int
    keypoints_frame: int
    keypoints_reference: int
    matched: bool
    reference_frame_path: str | None = None
    reason: str | None = None


class ORBViewMatcher:
    def __init__(
        self,
        *,
        nfeatures: int = 1200,
        ratio_test: float = 0.75,
        min_good_matches: int = 12,
        min_score: float = 0.20,
        max_dim: int = 640,
        homography_min_matches: int = 8,
    ) -> None:
        self.nfeatures = int(nfeatures)
        self.ratio_test = float(ratio_test)
        self.min_good_matches = int(min_good_matches)
        self.min_score = float(min_score)
        self.max_dim = int(max_dim)
        self.homography_min_matches = int(homography_min_matches)
        self._orb = cv2.ORB_create(nfeatures=self.nfeatures)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("frame is None")
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        height, width = gray.shape[:2]
        max_current = max(height, width)
        if max_current > self.max_dim:
            scale = self.max_dim / float(max_current)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray

    def score_pair(self, frame: np.ndarray, reference: np.ndarray) -> ViewMatchResult:
        current = self._prepare(frame)
        ref = self._prepare(reference)

        kp_current, des_current = self._orb.detectAndCompute(current, None)
        kp_ref, des_ref = self._orb.detectAndCompute(ref, None)

        keypoints_frame = len(kp_current or [])
        keypoints_reference = len(kp_ref or [])

        if des_current is None or des_ref is None:
            return ViewMatchResult(
                view_id=None,
                score=0.0,
                good_matches=0,
                keypoints_frame=keypoints_frame,
                keypoints_reference=keypoints_reference,
                matched=False,
                reason="no_descriptors",
            )

        knn_matches = self._bf.knnMatch(des_current, des_ref, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < self.ratio_test * second.distance:
                good_matches.append(best)

        good_count = len(good_matches)
        if good_count == 0:
            return ViewMatchResult(
                view_id=None,
                score=0.0,
                good_matches=0,
                keypoints_frame=keypoints_frame,
                keypoints_reference=keypoints_reference,
                matched=False,
                reason="no_good_matches",
            )

        match_density = good_count / max(1, min(keypoints_frame, keypoints_reference))
        quality = min(1.0, good_count / max(1, self.min_good_matches))

        inlier_ratio = 0.0
        if good_count >= self.homography_min_matches:
            src_pts = np.float32([kp_current[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None and mask.size > 0:
                inlier_ratio = float(mask.sum()) / max(1, good_count)

        score = 0.5 * match_density + 0.3 * quality + 0.2 * inlier_ratio
        score = float(max(0.0, min(1.0, score)))
        matched = score >= self.min_score and good_count >= self.min_good_matches

        return ViewMatchResult(
            view_id=None,
            score=score,
            good_matches=good_count,
            keypoints_frame=keypoints_frame,
            keypoints_reference=keypoints_reference,
            matched=matched,
            reason=None if matched else "below_threshold",
        )

    def match_cache(
        self,
        frame: np.ndarray,
        cache: SlotCache,
        *,
        reference_loader: Callable[[SlotView], np.ndarray | None] | None = None,
    ) -> ViewMatchResult:
        best: ViewMatchResult | None = None
        for view in cache.views:
            reference = None
            if reference_loader is not None:
                reference = reference_loader(view)
            if reference is None:
                continue
            result = self.score_pair(frame, reference)
            result = ViewMatchResult(
                view_id=view.id,
                score=result.score,
                good_matches=result.good_matches,
                keypoints_frame=result.keypoints_frame,
                keypoints_reference=result.keypoints_reference,
                matched=result.matched,
                reference_frame_path=view.reference_frame_path or None,
                reason=result.reason,
            )
            if best is None or result.score > best.score:
                best = result

        if best is None:
            return ViewMatchResult(
                view_id=None,
                score=0.0,
                good_matches=0,
                keypoints_frame=0,
                keypoints_reference=0,
                matched=False,
                reason="no_references",
            )
        return best


def load_reference_frame(path: str | Path) -> np.ndarray | None:
    if not path:
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    return image
