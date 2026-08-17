import os
from pathlib import Path
from typing import List, Dict, Any

from src.data.validator import DatasetValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AICityProcessor:
    """
    Dataset processor for AI City Challenge benchmark.
    """
    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir) / "aicity"

    def exists(self) -> bool:
        return self.raw_dir.exists() and len(list(self.raw_dir.rglob("*.jpg"))) > 0

    def process(self) -> List[Dict[str, Any]]:
        records = []
        if not self.exists():
            logger.info("AI City Challenge dataset not found in raw data directory.")
            return records

        img_paths = list(self.raw_dir.rglob("*.jpg")) + list(self.raw_dir.rglob("*.png"))
        for p in img_paths:
            if not DatasetValidator.is_valid_image(p):
                continue
            records.append({
                "image_path": str(p.resolve()),
                "dataset": "AI City Challenge",
                "manufacturer": "Generic",
                "model": "Generic",
                "vehicle_class": "car",
                "fuel_type": "UNKNOWN",
                "fuel_confidence": 0.0,
                "split": "train"
            })
        return records
