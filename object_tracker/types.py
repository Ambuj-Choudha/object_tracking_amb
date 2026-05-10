from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (cx, cy, w, h)
