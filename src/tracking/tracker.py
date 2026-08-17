import math
import cv2
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union, Any

from src.detection.types import BoundingBox, DetectionResult
from src.tracking.line_counter import VirtualCountingLine
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TrackedVehicle:
    """
    State object for an active vehicle track across video frames.
    """
    def __init__(
        self,
        track_id: int,
        initial_detection: DetectionResult,
        frame_idx: int = 0,
        timestamp: str = "",
        fps: float = 25.0
    ):
        self.track_id = track_id
        self.class_id = initial_detection.class_id
        self.class_name = initial_detection.class_name
        self.confidence = float(initial_detection.confidence)
        self.fuel_type = initial_detection.fuel_type or "Unknown"
        self.fuel_confidence = initial_detection.fuel_confidence or 0.50
        
        self.bbox = initial_detection.bbox
        self.vehicle_crop = initial_detection.vehicle_crop
        
        cx, cy = self.bbox.center
        self.center_x = float(cx)
        self.center_y = float(cy)
        
        self.first_frame_idx = frame_idx
        self.last_frame_idx = frame_idx
        self.first_timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.last_timestamp = self.first_timestamp
        
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.fps = fps
        self.speed_kmh: float = 0.0

        self.trajectory: List[Tuple[float, float]] = [(self.center_x, self.center_y)]

    @property
    def dwell_time_seconds(self) -> float:
        """Returns dwell time duration in seconds."""
        frames_active = max(1, self.last_frame_idx - self.first_frame_idx + 1)
        return float(frames_active / self.fps)

    def update(
        self,
        detection: DetectionResult,
        frame_idx: int,
        timestamp: str,
        px_to_meter: float = 0.05
    ):
        """Updates track attributes with new frame detection."""
        self.bbox = detection.bbox
        self.confidence = float(detection.confidence)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        self.last_frame_idx = frame_idx
        self.last_timestamp = timestamp

        if detection.vehicle_crop is not None:
            self.vehicle_crop = detection.vehicle_crop
        if detection.fuel_type and detection.fuel_type != "Unknown":
            self.fuel_type = detection.fuel_type
            self.fuel_confidence = detection.fuel_confidence or 0.80

        # Trajectory & Speed Filter
        cx, cy = self.bbox.center
        prev_cx, prev_cy = self.trajectory[-1]
        self.center_x = float(cx)
        self.center_y = float(cy)
        self.trajectory.append((self.center_x, self.center_y))
        if len(self.trajectory) > 50:
            self.trajectory.pop(0)

        # Distance speed calculation
        dist_px = math.hypot(self.center_x - prev_cx, self.center_y - prev_cy)
        dist_meters = dist_px * px_to_meter
        instant_speed = dist_meters * self.fps * 3.6  # m/s to km/h
        self.speed_kmh = 0.7 * self.speed_kmh + 0.3 * instant_speed

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1


class VehicleTracker:
    """
    Multi-Object Vehicle Tracker & Traffic Flow Analyzer.
    Maintains persistent IDs, unique vehicle counts, entry/exit virtual counting lines, 
    traffic density metrics, and dwell time estimations.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        fps: float = 25.0,
        counting_line: Optional[VirtualCountingLine] = None
    ):
        self.sys_config = config or load_config()
        trk_cfg = self.sys_config.get("tracking", {})
        
        self.max_age = trk_cfg.get("max_age", 30)
        self.min_hits = trk_cfg.get("min_hits", 3)
        self.iou_threshold = trk_cfg.get("iou_threshold", 0.30)
        self.px_to_meter = trk_cfg.get("pixel_to_meter_scale", 0.05)
        self.fps = fps

        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackedVehicle] = {}

        # Unique vehicle track history registry
        self.unique_track_history: Dict[int, Dict[str, Any]] = {}
        self.per_class_unique_counts: Dict[str, int] = {}
        
        # Virtual counting line
        self.counting_line = counting_line or VirtualCountingLine(name="Main Intersection Line")

        # Telemetry logs history
        self.telemetry_history: List[Dict[str, Any]] = []

    def _compute_iou(self, boxA: BoundingBox, boxB: BoundingBox) -> float:
        xA = max(boxA.x1, boxB.x1)
        yA = max(boxA.y1, boxB.y1)
        xB = min(boxA.x2, boxB.x2)
        yB = min(boxA.y2, boxB.y2)

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        unionArea = boxA.area + boxB.area - interArea
        return interArea / float(unionArea + 1e-6)

    def update(
        self,
        detections: List[DetectionResult],
        frame_idx: int = 0,
        timestamp: Optional[str] = None
    ) -> List[DetectionResult]:
        """
        Updates multi-object tracker state with current frame detections.
        Attaches persistent track_id to each detection object.
        
        Returns updated DetectionResult list.
        """
        ts = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        track_ids = list(self.active_tracks.keys())
        updated_detections: List[DetectionResult] = []

        if not track_ids:
            # Register new tracks for initial frame
            for det in detections:
                tid = self.next_track_id
                self.next_track_id += 1
                new_track = TrackedVehicle(tid, det, frame_idx=frame_idx, timestamp=ts, fps=self.fps)
                self.active_tracks[tid] = new_track
                
                det.track_id = tid
                updated_detections.append(det)

                self._register_unique_vehicle(new_track)
                self._record_telemetry(new_track, frame_idx, ts)

            return updated_detections

        # Match existing tracks to detections using IoU matrix
        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()

        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(self.active_tracks[tid].bbox, det.bbox)

        # Greedy matching
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break

            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            tid = track_ids[i]
            det = detections[j]

            track = self.active_tracks[tid]
            track.update(det, frame_idx=frame_idx, timestamp=ts, px_to_meter=self.px_to_meter)
            det.track_id = tid
            updated_detections.append(det)

            # Check virtual line crossing
            self.counting_line.check_crossing(tid, track.class_name, track.trajectory)

            matched_tracks.add(tid)
            matched_dets.add(j)

            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

            self._record_telemetry(track, frame_idx, ts)

        # Register un-matched detections as new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                new_track = TrackedVehicle(tid, det, frame_idx=frame_idx, timestamp=ts, fps=self.fps)
                self.active_tracks[tid] = new_track

                det.track_id = tid
                updated_detections.append(det)

                self._register_unique_vehicle(new_track)
                self._record_telemetry(new_track, frame_idx, ts)

        # Handle missed tracks & cleanup stale tracks
        stale_tids = []
        for tid, track in self.active_tracks.items():
            if tid not in matched_tracks:
                track.mark_missed()
                if track.time_since_update > self.max_age:
                    stale_tids.append(tid)

        for tid in stale_tids:
            del self.active_tracks[tid]

        return updated_detections

    def _register_unique_vehicle(self, track: TrackedVehicle):
        """Registers a unique vehicle track ID in historical records."""
        if track.track_id not in self.unique_track_history:
            v_cls = track.class_name.lower()
            self.unique_track_history[track.track_id] = {
                "track_id": track.track_id,
                "class_name": v_cls,
                "first_seen_frame": track.first_frame_idx,
                "first_seen_time": track.first_timestamp
            }
            self.per_class_unique_counts[v_cls] = self.per_class_unique_counts.get(v_cls, 0) + 1
            logger.info(f"Registered NEW unique vehicle #{track.track_id} [{v_cls.upper()}]. Total unique: {len(self.unique_track_history)}")

    def _record_telemetry(self, track: TrackedVehicle, frame_idx: int, timestamp: str):
        """Records per-frame telemetry dictionary into history."""
        x1, y1, x2, y2 = track.bbox.to_int_tuple()
        self.telemetry_history.append({
            "timestamp": timestamp,
            "frame": frame_idx,
            "track_id": track.track_id,
            "vehicle_class": track.class_name,
            "confidence": round(track.confidence, 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": round(track.center_x, 2),
            "center_y": round(track.center_y, 2),
            "speed_kmh": round(track.speed_kmh, 1),
            "dwell_time_sec": round(track.dwell_time_seconds, 2)
        })

    def get_active_density(self) -> Dict[str, Any]:
        """Calculates current traffic density metrics."""
        active_count = len(self.active_tracks)
        active_by_class: Dict[str, int] = {}
        for track in self.active_tracks.values():
            v_cls = track.class_name.lower()
            active_by_class[v_cls] = active_by_class.get(v_cls, 0) + 1

        return {
            "active_vehicles_count": active_count,
            "active_by_class": active_by_class,
            "total_unique_vehicles": len(self.unique_track_history),
            "per_class_unique": self.per_class_unique_counts,
            "line_crossing_counts": self.counting_line.get_counts()
        }

    def export_tracking_results(self, output_csv_path: Union[str, Path]):
        """
        Exports full tracking history to tracking_results.csv format matching columns:
        timestamp, frame, track_id, vehicle_class, confidence, x1, y1, x2, y2, center_x, center_y
        """
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.telemetry_history)
        df.to_csv(out_p, index=False)
        logger.info(f"Saved tracking results ({len(df)} records) to {out_p}")

    def draw_tracking_overlay(
        self,
        frame: np.ndarray,
        show_counting_line: bool = True,
        show_trajectories: bool = True
    ) -> np.ndarray:
        """
        Renders tracking bounding boxes, track IDs, trajectory trails, 
        virtual counting line, and real-time summary total banner onto video frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Virtual Counting Line
        if show_counting_line and self.counting_line:
            p1 = (int(self.counting_line.line_start[0]), int(self.counting_line.line_start[1]))
            p2 = (int(self.counting_line.line_end[0]), int(self.counting_line.line_end[1]))
            cv2.line(annotated, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(annotated, f"COUNTING LINE: {self.counting_line.total_line_crossings}",
                        (p1[0] + 10, max(20, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 2. Draw Active Tracks & Trajectories
        for tid, track in self.active_tracks.items():
            x1, y1, x2, y2 = track.bbox.to_int_tuple()
            cls_name = track.class_name.upper()
            
            # Color palette
            color = (0, 255, 0) if "car" in cls_name.lower() else ((255, 100, 0) if "truck" in cls_name.lower() else (0, 215, 255))
            
            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label banner
            lbl = f"#{tid} {cls_name} ({int(track.confidence*100)}%)"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, lbl, (x1 + 4, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Center point
            cv2.circle(annotated, (int(track.center_x), int(track.center_y)), 4, (0, 0, 255), -1)

            # Trajectory trails
            if show_trajectories and len(track.trajectory) > 1:
                pts = np.array(track.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=(255, 255, 0), thickness=2)

        # 3. Top Banner Stats Summary
        cv2.rectangle(annotated, (0, 0), (w, 40), (20, 20, 20), -1)
        banner_txt = f"TOTAL UNIQUE VEHICLES: {len(self.unique_track_history)}  |  ACTIVE: {len(self.active_tracks)}  |  LINE CROSSINGS: {self.counting_line.total_line_crossings}"
        cv2.putText(annotated, banner_txt, (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated
