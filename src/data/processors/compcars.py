import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.data.fuel_lookup import FuelLookupManager
from src.data.validator import DatasetValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CompCarsProcessor:
    """
    Dataset processor for CompCars (Comprehensive Car Dataset).
    """
    def __init__(self, raw_dir: Path, fuel_lookup: Optional[FuelLookupManager] = None):
        self.raw_dir = Path(raw_dir)
        self.fuel_lookup = fuel_lookup or FuelLookupManager()

    def exists(self) -> bool:
        img_dir = self.raw_dir / "image"
        return self.raw_dir.exists() and (img_dir.exists() or len(list(self.raw_dir.rglob("*.jpg"))) > 0)

    def process(self) -> List[Dict[str, Any]]:
        records = []
        if not self.exists():
            logger.info("CompCars dataset not found in raw data directory.")
            return records

        img_paths = list(self.raw_dir.rglob("*.jpg")) + list(self.raw_dir.rglob("*.png"))
        logger.info(f"Found {len(img_paths)} candidate image files in CompCars directory.")

        for p in img_paths:
            if not DatasetValidator.is_valid_image(p):
                continue

            # CompCars structure: image/make_id/model_id/year/xxx.jpg
            rel_parts = p.relative_to(self.raw_dir).parts
            if len(rel_parts) >= 3:
                mfr = f"Make_{rel_parts[1]}"
                model = f"Model_{rel_parts[2]}"
            else:
                mfr = "Unknown"
                model = "Unknown"

            fuel_type, confidence = self.fuel_lookup.lookup(mfr, model)

            records.append({
                "image_path": str(p.resolve()),
                "dataset": "CompCars",
                "manufacturer": mfr,
                "model": model,
                "vehicle_class": "car",
                "fuel_type": fuel_type,
                "fuel_confidence": confidence,
                "split": "train" if hash(p.name) % 10 < 8 else "test"
            })

        return records
