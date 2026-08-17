import os
import pytest
import pandas as pd
from pathlib import Path

from src.detection.types import BoundingBox, DetectionResult
from src.tracking.tracker import VehicleTracker, TrackedVehicle
from src.tracking.line_counter import VirtualCountingLine, line_intersects

def test_line_intersection():
    # Segment 1: (0, 10) to (10, 10)
    # Segment 2: (5, 0) to (5, 20)
    s1 = ((0.0, 10.0), (10.0, 10.0))
    s2 = ((5.0, 0.0), (5.0, 20.0))
    assert line_intersects(s1, s2)

    # Non-intersecting parallel segments
    s3 = ((0.0, 5.0), (10.0, 5.0))
    assert not line_intersects(s1, s3)

def test_virtual_counting_line():
    line = VirtualCountingLine(line_start=(0, 100), line_end=(500, 100), name="Test Line")
    
    # Trajectory crossing the line y=100
    traj = [(250.0, 80.0), (250.0, 120.0)]
    crossed = line.check_crossing(track_id=1, vehicle_class="car", trajectory=traj)
    assert crossed
    assert line.total_line_crossings == 1

    # Second check for same track_id should NOT increment (no double counting!)
    crossed_again = line.check_crossing(track_id=1, vehicle_class="car", trajectory=traj)
    assert not crossed_again
    assert line.total_line_crossings == 1

def test_persistent_tracking_ids():
    tracker = VehicleTracker(fps=25.0)
    
    det1 = DetectionResult(
        bbox=BoundingBox(100, 100, 200, 200),
        class_id=2,
        class_name="car",
        confidence=0.90
    )
    
    # Frame 0
    res0 = tracker.update([det1], frame_idx=0)
    assert len(res0) == 1
    assert res0[0].track_id == 1

    # Frame 1 (slight movement)
    det2 = DetectionResult(
        bbox=BoundingBox(105, 105, 205, 205),
        class_id=2,
        class_name="car",
        confidence=0.92
    )
    res1 = tracker.update([det2], frame_idx=1)
    assert len(res1) == 1
    assert res1[0].track_id == 1  # ID preserved

    # Unique vehicle count must be EXACTLY 1, NOT 2
    density = tracker.get_active_density()
    assert density["total_unique_vehicles"] == 1

def test_dwell_time_estimation():
    det = DetectionResult(bbox=BoundingBox(10, 10, 50, 50), class_id=2, class_name="car", confidence=0.85)
    track = TrackedVehicle(track_id=1, initial_detection=det, frame_idx=0, fps=25.0)
    
    # Update across 25 frames (1 second)
    for f in range(1, 26):
        det_f = DetectionResult(bbox=BoundingBox(10+f, 10, 50+f, 50), class_id=2, class_name="car", confidence=0.85)
        track.update(det_f, frame_idx=f, timestamp="2026-08-12 12:00:00.000")

    assert pytest.approx(track.dwell_time_seconds, 0.05) == 1.04  # ~1 second

def test_tracking_csv_export(tmp_path):
    tracker = VehicleTracker(fps=25.0)
    det = DetectionResult(bbox=BoundingBox(10, 20, 110, 120), class_id=2, class_name="car", confidence=0.89)
    tracker.update([det], frame_idx=0)
    
    csv_file = tmp_path / "tracking_results.csv"
    tracker.export_tracking_results(csv_file)
    
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    expected_cols = ["timestamp", "frame", "track_id", "vehicle_class", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
    for col in expected_cols:
        assert col in df.columns
