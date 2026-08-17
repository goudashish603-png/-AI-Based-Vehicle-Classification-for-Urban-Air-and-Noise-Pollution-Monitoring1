import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

class YOLOConverter:
    """
    Utility to convert object detection bounding boxes to standard YOLO txt format.
    Generates YOLO dataset.yaml configuration files.
    """
    
    @staticmethod
    def bbox_to_yolo(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_w: int,
        img_h: int
    ) -> Tuple[float, float, float, float]:
        """
        Converts absolute box (x1, y1, x2, y2) to normalized YOLO (xc, yc, w, h).
        """
        dw = 1.0 / max(1, img_w)
        dh = 1.0 / max(1, img_h)

        xc = (x1 + x2) / 2.0 * dw
        yc = (y1 + y2) / 2.0 * dh
        w = (x2 - x1) * dw
        h = (y2 - y1) * dh

        # Clip bounds to [0.0, 1.0]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        return xc, yc, w, h

    @staticmethod
    def save_yolo_txt(
        output_txt_path: Path,
        annotations: List[Tuple[int, float, float, float, float]]
    ):
        """
        Saves a list of (class_id, xc, yc, w, h) tuples to a YOLO text file.
        """
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt_path, "w") as f:
            for cls_id, xc, yc, w, h in annotations:
                f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    @staticmethod
    def create_yolo_dataset_yaml(
        dataset_root: Path,
        train_path: str,
        val_path: str,
        class_names: Dict[int, str],
        output_yaml_path: Path
    ):
        """
        Generates standard YOLO dataset.yaml file.
        """
        yaml_content = {
            "path": str(dataset_root.resolve()),
            "train": train_path,
            "val": val_path,
            "names": class_names
        }
        output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        logger.info(f"Generated YOLO dataset config at {output_yaml_path}")
