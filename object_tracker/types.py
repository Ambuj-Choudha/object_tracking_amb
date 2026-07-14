from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
