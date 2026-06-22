"""Frame anonymization utilities for runtime privacy masking."""

from .opencv import OpenCVHaarAnonymizer, Region

__all__ = [
    "OpenCVHaarAnonymizer",
    "Region",
]
