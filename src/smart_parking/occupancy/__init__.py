"""Occupancy engine for canonical parking-slot polygons."""

from .engine import (
    NormalizedSlot,
    OccupancyEngine,
    OccupancyResult,
    SlotOccupancyStatus,
    normalize_slots,
)

__all__ = [
    "NormalizedSlot",
    "OccupancyEngine",
    "OccupancyResult",
    "SlotOccupancyStatus",
    "normalize_slots",
]
