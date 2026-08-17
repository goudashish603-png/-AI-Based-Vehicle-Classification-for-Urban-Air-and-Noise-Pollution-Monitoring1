import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.data.validator import DatasetValidator
from src.data.fuel_lookup import FuelLookupManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SampleDataProcessor:
    """
    Processor for local synthetic/sample traffic images and videos.
    Ensures the pipeline is fully executable out-of-the-box.
    """
    def __init__(self, raw_dir: Path, fuel_lookup: Optional[FuelLookupManager] = None):
        self.raw_dir = Path(raw_dir)
        self.fuel_lookup = fuel_lookup or FuelLookupManager()

    def exists(self) -> bool:
        img_dir = self.raw_dir / "images"
        return self.raw_dir.exists() and img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 0

    def process(self) -> List[Dict[str, Any]]:
        records = []
        if not self.exists():
            return records

        img_dir = self.raw_dir / "images"
        img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        logger.info(f"Processing {len(img_paths)} sample traffic images...")

        sample_cars = [
            ("Toyota", "Prius", "HYBRID"),
            ("Tesla", "Model 3", "EV"),
            ("BMW", "3 Series", "PETROL"),
            ("Ford", "F-150", "DIESEL"),
            ("Maruti Suzuki", "Wagon R", "CNG_LPG")
        ]

        for i, p in enumerate(img_paths):
            if not DatasetValidator.is_valid_image(p):
                continue

            mfr, model, fuel = sample_cars[i % len(sample_cars)]

            records.append({
                "image_path": str(p.resolve()),
                "dataset": "Sample Dataset",
                "manufacturer": mfr,
                "model": model,
                "vehicle_class": "car" if "car" in p.name.lower() or i % 2 == 0 else "bus",
                "fuel_type": fuel,
                "fuel_confidence": 0.95,
                "split": "train" if i % 4 != 0 else "test"
            })

        return records
