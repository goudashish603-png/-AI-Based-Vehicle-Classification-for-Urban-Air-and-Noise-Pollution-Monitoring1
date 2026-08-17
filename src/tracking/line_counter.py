import numpy as np
from typing import Tuple, List, Dict, Set, Optional
from pathlib import Path

from src.detection.types import BoundingBox
from src.utils.logger import get_logger

logger = get_logger(__name__)

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Helper to check if three points A, B, C are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_intersects(
    segment1: Tuple[Tuple[float, float], Tuple[float, float]],
    segment2: Tuple[Tuple[float, float], Tuple[float, float]]
) -> bool:
    """
    Checks if two line segments (A-B and C-D) intersect in 2D space.
    """
    A, B = segment1
    C, D = segment2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class VirtualCountingLine:
    """
    Virtual Counting Line for Traffic Flow Monitoring.
    Detects when vehicle trajectories cross a defined 2D line segment.
    """
    def __init__(
        self,
        line_start: Tuple[int, int] = (0, 360),
        line_end: Tuple[int, int] = (1280, 360),
        name: str = "Main Traffic Line"
    ):
        self.line_start = (float(line_start[0]), float(line_start[1]))
        self.line_end = (float(line_end[0]), float(line_end[1]))
        self.name = name
        
        self.counted_track_ids: Set[int] = set()
        self.counts_by_class: Dict[str, int] = {}
        self.total_line_crossings: int = 0

    def check_crossing(
        self,
        track_id: int,
        vehicle_class: str,
        trajectory: List[Tuple[float, float]]
    ) -> bool:
        """
        Checks if vehicle trajectory crossed the virtual line.
        Ensures each track_id is counted AT MOST ONCE.
        
        Returns:
            True if vehicle just crossed line on this frame, False otherwise.
        """
        if track_id in self.counted_track_ids:
            return False

        if len(trajectory) < 2:
            return False

        prev_pt = trajectory[-2]
        curr_pt = trajectory[-1]

        trajectory_segment = (prev_pt, curr_pt)
        counting_segment = (self.line_start, self.line_end)

        if line_intersects(trajectory_segment, counting_segment):
            self.counted_track_ids.add(track_id)
            self.total_line_crossings += 1
            v_cls = vehicle_class.lower()
            self.counts_by_class[v_cls] = self.counts_by_class.get(v_cls, 0) + 1
            logger.info(f"Vehicle #{track_id} ({vehicle_class}) crossed counting line '{self.name}'. Total: {self.total_line_crossings}")
            return True

        return False

    def get_counts(self) -> Dict[str, int]:
        """Returns per-class line crossing counts."""
        res = dict(self.counts_by_class)
        res["total"] = self.total_line_crossings
        return res
