from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class DetectionResult:
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    vehicle_crop: Optional[np.ndarray] = None
    fuel_type: Optional[str] = None
    fuel_confidence: Optional[float] = None
