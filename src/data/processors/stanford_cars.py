import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.data.fuel_lookup import FuelLookupManager
from src.data.validator import DatasetValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StanfordCarsProcessor:
    """
    Dataset processor for Stanford Cars fine-grained vehicle dataset.
    """
    def __init__(self, raw_dir: Path, fuel_lookup: Optional[FuelLookupManager] = None):
        self.raw_dir = Path(raw_dir)
        self.fuel_lookup = fuel_lookup or FuelLookupManager()

    def exists(self) -> bool:
        """Checks if Stanford Cars dataset exists in data/raw/stanford_cars."""
        img_dir = self.raw_dir / "car_ims"
        return self.raw_dir.exists() and (img_dir.exists() or len(list(self.raw_dir.glob("*.jpg"))) > 0)

    def process(self) -> List[Dict[str, Any]]:
        """
        Parses dataset files, validates images, maps manufacturer/model to fuel type.
        
        Returns list of metadata dicts.
        """
        records = []
        if not self.exists():
            logger.info("Stanford Cars dataset not found in raw data directory.")
            return records

        # Scan image files
        img_paths = list(self.raw_dir.rglob("*.jpg")) + list(self.raw_dir.rglob("*.png"))
        logger.info(f"Found {len(img_paths)} candidate image files in Stanford Cars directory.")

        for p in img_paths:
            if not DatasetValidator.is_valid_image(p):
                logger.warning(f"Corrupt or unreadable image skipped: {p}")
                continue

            # Parse filename or folder structure for make/model
            # Example filename: "00001_BMW_3_Series_Sedan_2012.jpg" or folder "BMW 3 Series"
            stem = p.stem.replace("_", " ")
            parts = stem.split()

            mfr = parts[0] if parts else "Unknown"
            model = " ".join(parts[1:3]) if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else "Unknown")
            
            # Vehicle class heuristic
            vcls = "car"
            if "SUV" in stem or "Truck" in stem:
                vcls = "truck"
            elif "Van" in stem or "Minivan" in stem:
                vcls = "car"

            # Fuel Lookup
            fuel_type, confidence = self.fuel_lookup.lookup(mfr, model)

            records.append({
                "image_path": str(p.resolve()),
                "dataset": "Stanford Cars",
                "manufacturer": mfr,
                "model": model,
                "vehicle_class": vcls,
                "fuel_type": fuel_type,
                "fuel_confidence": confidence,
                "split": "train" if hash(p.name) % 10 < 8 else "test"
            })

        return records
