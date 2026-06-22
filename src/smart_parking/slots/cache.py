from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .geometry import coerce_image_shape, normalize_slots


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SlotView:
    id: str
    created_at: str | None
    image_shape: tuple[int, int] | None
    reference_frame_path: str
    slots: list[dict[str, Any]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "image_shape": list(self.image_shape) if self.image_shape is not None else None,
            "reference_frame_path": self.reference_frame_path,
            "slots": deepcopy(self.slots),
        }


@dataclass
class SlotCache:
    cache_path: Path
    version: int = 1
    views: list[SlotView] = field(default_factory=list)

    def get_view(self, view_id: str | None) -> SlotView | None:
        if view_id is None:
            return None
        for view in self.views:
            if view.id == view_id:
                return view
        return None

    def resolve_reference_frame_path(self, view: SlotView | None) -> Path | None:
        if view is None or not view.reference_frame_path:
            return None
        return (self.cache_path.parent / view.reference_frame_path).resolve()

    def upsert_view(self, view: SlotView) -> None:
        for index, existing in enumerate(self.views):
            if existing.id == view.id:
                self.views[index] = view
                return
        self.views.append(view)

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "views": [view.to_payload() for view in self.views],
        }


def canonicalize_slots(
    slots: Sequence[Mapping[str, Any]] | None,
    image_shape: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    return normalize_slots(slots, image_shape=image_shape)


def build_view_payload(
    *,
    view_id: str,
    image_shape: tuple[int, int] | list[int] | None,
    reference_frame_path: str,
    slots: Sequence[Mapping[str, Any]],
    created_at: str | None = None,
) -> dict[str, Any]:
    normalized_shape = coerce_image_shape(image_shape)
    return {
        "id": str(view_id),
        "created_at": created_at or _iso_now(),
        "image_shape": list(normalized_shape) if normalized_shape is not None else None,
        "reference_frame_path": str(reference_frame_path),
        "slots": canonicalize_slots(slots, image_shape=normalized_shape),
    }


def _payload_to_view(payload: Mapping[str, Any], *, default_id: str) -> SlotView:
    image_shape = coerce_image_shape(payload.get("image_shape"))
    slots = canonicalize_slots(payload.get("slots", []), image_shape=image_shape)
    return SlotView(
        id=str(payload.get("id", default_id)),
        created_at=payload.get("created_at"),
        image_shape=image_shape,
        reference_frame_path=str(payload.get("reference_frame_path", "")),
        slots=slots,
    )


def _legacy_slots_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        return {
            "image_shape": None,
            "slots": canonicalize_slots(data),
        }
    if isinstance(data, dict) and "slots" in data and isinstance(data["slots"], list):
        image_shape = coerce_image_shape(data.get("image_shape"))
        return {
            "image_shape": list(image_shape) if image_shape is not None else None,
            "slots": canonicalize_slots(data["slots"], image_shape=image_shape),
        }
    raise ValueError("Unsupported slot payload")


def load_slot_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and payload.get("version") == 1 and "views" in payload:
        return payload
    return _legacy_slots_payload(payload)


def load_view_cache(path: str | Path) -> SlotCache:
    cache_path = Path(path)
    if not cache_path.exists():
        return SlotCache(cache_path=cache_path, version=1, views=[])

    payload = load_slot_file(cache_path)
    if payload.get("version") == 1 and "views" in payload:
        views = [
            _payload_to_view(view_payload, default_id=f"view-{index + 1:04d}")
            for index, view_payload in enumerate(payload.get("views", []))
            if isinstance(view_payload, Mapping)
        ]
        return SlotCache(
            cache_path=cache_path,
            version=int(payload.get("version", 1)),
            views=views,
        )

    legacy_shape = coerce_image_shape(payload.get("image_shape"))
    legacy_view = SlotView(
        id="view-0001",
        created_at=None,
        image_shape=legacy_shape,
        reference_frame_path="",
        slots=canonicalize_slots(payload.get("slots", []), image_shape=legacy_shape),
    )
    return SlotCache(cache_path=cache_path, version=1, views=[legacy_view])


def save_view_cache(path: str | Path, cache: SlotCache | dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cache, SlotCache):
        payload = cache.to_payload()
    else:
        payload = cache
    with cache_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def select_view_payload(cache: SlotCache | dict[str, Any], view_id: str | None = None) -> dict[str, Any] | None:
    if isinstance(cache, SlotCache):
        view = cache.get_view(view_id) if view_id is not None else (cache.views[0] if cache.views else None)
        return view.to_payload() if view is not None else None

    views = cache.get("views", [])
    if not views:
        return None
    if view_id is None:
        return deepcopy(views[0])
    for view in views:
        if view.get("id") == view_id:
            return deepcopy(view)
    return None
