import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "external" / "emission_factors.csv"

SUPPORTED_POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO", "SO2", "CO2"]

# Default fallback factors (g/km) derived from EEA COPERT V fleet averages
DEFAULT_FALLBACK_FACTORS = {
    "PM2.5": 0.015,
    "PM10": 0.020,
    "NO2": 0.150,
    "CO": 0.500,
    "SO2": 0.003,
    "CO2": 160.0
}

class EmissionFactorRegistry:
    """
    Registry for vehicle emission factors (g/km) sourced from EEA COPERT V standards.
    Reads data/external/emission_factors.csv.
    """
    def __init__(self, csv_path: Optional[Union[str, Path]] = None):
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        self.factors_df = pd.DataFrame()
        self._lookup_table: Dict[Tuple[str, str, str], float] = {}
        self.load_factors()

    def load_factors(self):
        """Loads and indexes emission factors from CSV database."""
        if not self.csv_path.exists():
            logger.warning(f"Emission factor database missing at {self.csv_path}. Initializing default registry.")
            self._create_empty_csv()

        try:
            df = pd.read_csv(self.csv_path)
            for _, row in df.iterrows():
                v_type = str(row["vehicle_type"]).strip().lower()
                f_type = str(row["fuel_type"]).strip().upper()
                pol = str(row["pollutant"]).strip()
                factor = float(row["emission_factor"]) if pd.notna(row["emission_factor"]) else 0.0

                self._lookup_table[(v_type, f_type, pol)] = factor

            self.factors_df = df
            logger.info(f"Loaded {len(self._lookup_table)} emission factors from {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading emission factor CSV: {e}")

    def _create_empty_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["vehicle_type", "fuel_type", "pollutant", "emission_factor", "unit", "source", "source_url", "notes"]
        pd.DataFrame(columns=cols).to_csv(self.csv_path, index=False)

    def get_factor(
        self,
        vehicle_type: str,
        fuel_type: str,
        pollutant: str
    ) -> float:
        """
        Retrieves emission factor in g/km for a given vehicle type, fuel type, and pollutant.
        Returns default fallback if specific entry is missing.
        """
        v_key = str(vehicle_type).strip().lower()
        f_key = str(fuel_type).strip().upper()
        p_key = str(pollutant).strip()

        # 1. Exact match (vehicle_type, fuel_type, pollutant)
        if (v_key, f_key, p_key) in self._lookup_table:
            return self._lookup_table[(v_key, f_key, p_key)]

        # 2. Match with fuel_type="UNKNOWN"
        if (v_key, "UNKNOWN", p_key) in self._lookup_table:
            return self._lookup_table[(v_key, "UNKNOWN", p_key)]

        # 3. Match with vehicle_type="car"
        if ("car", f_key, p_key) in self._lookup_table:
            return self._lookup_table[("car", f_key, p_key)]

        # 4. Fallback factor
        return DEFAULT_FALLBACK_FACTORS.get(p_key, 0.01)
