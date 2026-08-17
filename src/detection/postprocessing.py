import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union

from src.detection.types import DetectionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard vehicle class mapping
COCO_VEHICLE_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "other_vehicle",  # bicycle
    6: "train"           # mapped to other_vehicle
}

TARGET_VEHICLE_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "van",
    "other_vehicle"
]

def map_coco_class_name(class_id: int, original_name: str = "") -> str:
    """Normalizes class ID and name to target vehicle class set."""
    lower_name = str(original_name).lower().strip()
    if "van" in lower_name:
        return "van"
    elif "car" in lower_name or class_id == 2:
        return "car"
    elif "bus" in lower_name or class_id == 5:
        return "bus"
    elif "truck" in lower_name or class_id == 7:
        return "truck"
    elif "motorcycle" in lower_name or "bike" in lower_name or class_id == 3:
        return "motorcycle"
    else:
        return "other_vehicle"

def export_detections_to_csv(detections_records: List[Dict[str, Any]], output_path: Union[str, Path]):
    """Exports a list of detection dictionary records to CSV format."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(detections_records)
    df.to_csv(out_p, index=False)
    logger.info(f"Exported {len(df)} vehicle detection records to CSV: {out_p}")

def export_detections_to_json(detections_records: List[Dict[str, Any]], output_path: Union[str, Path]):
    """Exports a list of detection dictionary records to JSON format."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(detections_records, f, indent=2)
    logger.info(f"Exported {len(detections_records)} vehicle detection records to JSON: {out_p}")
