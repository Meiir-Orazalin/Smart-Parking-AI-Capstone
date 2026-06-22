"""Slot schema, cache, matching, and model-backed slot detection."""

from .cache import (
    SlotCache,
    SlotView,
    build_view_payload,
    canonicalize_slots,
    load_slot_file,
    load_view_cache,
    save_view_cache,
    select_view_payload,
)
from .geometry import bbox_from_points, points_from_bbox, scale_slots
from .generator import (
    SeedValidationError,
    default_preview_path,
    generate_slots_from_seed_file,
    generate_slots_from_seed_payload,
    load_anchor_seed_file,
    render_generated_slots_preview,
    save_anchor_seed_file,
)
from .matching import ORBViewMatcher, ViewMatchResult
from .model_detector import YOLOSlotPolygonDetector
from .state import FrameStabilityTracker, ViewStateManager, ViewStateSnapshot

__all__ = [
    "ORBViewMatcher",
    "FrameStabilityTracker",
    "SlotCache",
    "SlotView",
    "SeedValidationError",
    "ViewMatchResult",
    "ViewStateManager",
    "ViewStateSnapshot",
    "YOLOSlotPolygonDetector",
    "bbox_from_points",
    "build_view_payload",
    "canonicalize_slots",
    "default_preview_path",
    "generate_slots_from_seed_file",
    "generate_slots_from_seed_payload",
    "load_slot_file",
    "load_anchor_seed_file",
    "load_view_cache",
    "points_from_bbox",
    "render_generated_slots_preview",
    "save_view_cache",
    "save_anchor_seed_file",
    "scale_slots",
    "select_view_payload",
]
