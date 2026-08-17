import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.detection.types import DetectionResult

# Color Palette for Fuel Badges (BGR format)
FUEL_COLOR_MAP = {
    "Petrol": (255, 191, 0),             # Deep Cyan / Amber
    "Diesel": (34, 34, 220),              # Deep Red
    "Electric Vehicle (EV)": (50, 205, 50),# Neon Green
    "CNG/LPG": (0, 215, 255),            # Bright Yellow
    "Hybrid": (211, 85, 186),             # Purple / Magenta
    "Unknown": (128, 128, 128)            # Grey
}

class VisualAnnotator:
    """
    OpenCV Frame Annotator for bounding boxes, fuel tags, speed, trajectory trails, and HUD dashboard overlays.
    """
    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[DetectionResult],
        show_trajectories: bool = True
    ) -> np.ndarray:
        """
        Draws bounding boxes, fuel badges, and track details onto video frames.
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_int_tuple()
            fuel = det.fuel_type or "Unknown"
            color = FUEL_COLOR_MAP.get(fuel, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label text: e.g. "#1 Car | Diesel (85%)"
            tid_str = f"#{det.track_id} " if det.track_id else ""
            fuel_conf_str = f" ({int((det.fuel_confidence or 0.8)*100)}%)" if det.fuel_confidence else ""
            label = f"{tid_str}{det.class_name.upper()} | {fuel}{fuel_conf_str}"

            # Label background banner
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + w + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    @staticmethod
    def draw_hud_overlay(
        frame: np.ndarray,
        vehicle_count: int,
        est_pm25_g_hr: float,
        est_nox_g_hr: float,
        noise_dba: float
    ) -> np.ndarray:
        """
        Draws a sleek Heads-Up Display (HUD) banner at top of frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Transparent black top panel
        panel_h = 50
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Telemetry text
        hud_text = (
            f"VEHICLES: {vehicle_count}  |  "
            f"EST. PM2.5: {est_pm25_g_hr:.2f} g/hr  |  "
            f"EST. NOx: {est_nox_g_hr:.2f} g/hr  |  "
            f"NOISE INDEX: {noise_dba:.1f} dBA"
        )
        cv2.putText(annotated, hud_text, (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)
        
        return annotated
