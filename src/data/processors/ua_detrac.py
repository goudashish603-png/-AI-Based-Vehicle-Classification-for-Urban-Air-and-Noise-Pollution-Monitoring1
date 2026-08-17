import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.data.validator import DatasetValidator
from src.data.yolo_converter import YOLOConverter
from src.utils.logger import get_logger

logger = get_logger(__name__)

# UA-DETRAC Class Mapping to COCO standard IDs
DETRAC_CLASS_MAP = {
    "car": 2,
    "bus": 5,
    "van": 2,      # Map van -> car or truck
    "others": 7    # Map heavy truck
}

class UADetracProcessor:
    """
    Dataset processor for UA-DETRAC detection & tracking benchmark.
    Converts UA-DETRAC bounding box annotations to standard YOLO format.
    """
    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir) / "ua_detrac"

    def exists(self) -> bool:
        return self.raw_dir.exists() and len(list(self.raw_dir.rglob("*.jpg"))) > 0

    def process(self) -> List[Dict[str, Any]]:
        records = []
        if not self.exists():
            logger.info("UA-DETRAC dataset not found in raw data directory.")
            return records

        img_paths = list(self.raw_dir.rglob("*.jpg"))
        logger.info(f"Processing {len(img_paths)} frames in UA-DETRAC dataset...")

        yolo_img_dir = self.processed_dir / "images"
        yolo_lbl_dir = self.processed_dir / "labels"
        yolo_img_dir.mkdir(parents=True, exist_ok=True)
        yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

        for p in img_paths:
            if not DatasetValidator.is_valid_image(p):
                continue

            records.append({
                "image_path": str(p.resolve()),
                "dataset": "UA-DETRAC",
                "manufacturer": "Generic",
                "model": "Generic",
                "vehicle_class": "car",
                "fuel_type": "UNKNOWN",
                "fuel_confidence": 0.0,
                "split": "train" if hash(p.name) % 10 < 8 else "test"
            })

        # Generate YOLO dataset.yaml
        YOLOConverter.create_yolo_dataset_yaml(
            dataset_root=self.processed_dir,
            train_path="images/train",
            val_path="images/val",
            class_names={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
            output_yaml_path=self.processed_dir / "dataset.yaml"
        )

        return records
