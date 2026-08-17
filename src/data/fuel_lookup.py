import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Set

from src.fuel_mapping.fuel_mapper import FuelTypeMapper, ALLOWED_FUEL_TYPES
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "external" / "fuel_mapping.csv"

class FuelLookupManager:
    """
    Manager for vehicle manufacturer/model to fuel-type mapping lookup table.
    Enforces strict fuel validation and wraps FuelTypeMapper.
    """
    def __init__(self, mapping_csv: Optional[Path] = None):
        self.csv_path = Path(mapping_csv) if mapping_csv else DEFAULT_MAPPING_PATH
        self.mapper = FuelTypeMapper(csv_path=self.csv_path)

    def lookup(self, manufacturer: str, model: str, variant: str = "") -> Tuple[str, float]:
        """
        Looks up fuel type and confidence score for a vehicle manufacturer/model/variant.
        """
        res = self.mapper.infer_fuel_type(manufacturer, model, variant)
        return res.fuel_type, res.confidence

    def update_mapping(
        self,
        manufacturer: str,
        model: str,
        variant: str,
        fuel_type: str,
        confidence: float = 0.95,
        source: str = "Manual Admin",
        source_url: str = "",
        notes: str = ""
    ):
        """
        Updates or inserts a vehicle fuel mapping record into the CSV database.
        """
        fuel_upper = str(fuel_type).strip().upper()
        if fuel_upper not in ALLOWED_FUEL_TYPES:
            raise ValueError(f"Invalid fuel_type '{fuel_type}'. Must be one of {ALLOWED_FUEL_TYPES}")

        df = self.mapper.mapping_df
        mask = (
            (df["manufacturer"].astype(str).str.lower() == manufacturer.lower()) &
            (df["model"].astype(str).str.lower() == model.lower()) &
            (df["variant"].astype(str).str.lower() == variant.lower())
        )

        if mask.any():
            df.loc[mask, "fuel_type"] = fuel_upper
            df.loc[mask, "confidence"] = confidence
            df.loc[mask, "source"] = source
            df.loc[mask, "source_url"] = source_url
            df.loc[mask, "notes"] = notes
        else:
            new_row = pd.DataFrame([{
                "manufacturer": manufacturer,
                "model": model,
                "variant": variant,
                "fuel_type": fuel_upper,
                "confidence": confidence,
                "source": source,
                "source_url": source_url,
                "notes": notes
            }])
            df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(self.csv_path, index=False)
        self.mapper.load_database()
