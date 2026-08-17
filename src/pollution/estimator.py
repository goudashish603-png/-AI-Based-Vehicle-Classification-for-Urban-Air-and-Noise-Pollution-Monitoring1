from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.utils.config import load_emissions_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class PollutionEstimate:
    pm2_5_g_per_hr: float
    pm10_g_per_hr: float
    nox_g_per_hr: float
    co_g_per_hr: float
    co2_g_per_hr: float
    total_vehicles: int
    fuel_counts: Dict[str, int]
    category_counts: Dict[str, int]

    @property
    def air_quality_index_contribution(self) -> float:
        """
        Computes an AI-Estimated Vehicle Pollution Contribution Score (0–100 scale).
        Note: This represents estimated traffic contribution, NOT direct chemical sensor concentration.
        """
        # Weighted combination of key pollutants relative to standard urban baseline
        raw_score = (self.pm2_5_g_per_hr * 12.0) + (self.nox_g_per_hr * 1.5) + (self.co_g_per_hr * 0.4)
        return min(100.0, max(0.0, raw_score))


class AirPollutionEstimator:
    """
    Mathematical Air Emission Estimation Model (COPERT V / EEA Air Pollutant Standard).
    Calculates Estimated Vehicle Pollution Contribution based on traffic counts, speeds, and fuel types.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.emissions_cfg = config or load_emissions_config()
        self.factors = self.emissions_cfg.get("emission_factors", {})

    def estimate_emissions(
        self,
        vehicle_records: List[Dict[str, Any]],
        duration_seconds: float = 1.0
    ) -> PollutionEstimate:
        """
        Calculates emission contribution for a set of vehicle records.
        
        Args:
            vehicle_records: List of dicts with keys: 'class_name', 'fuel_type', 'speed_kmh'
            duration_seconds: Time interval over which detection occurred
            
        Returns:
            PollutionEstimate object containing calculated mass rates (g/hr)
        """
        total_pm25 = 0.0
        total_pm10 = 0.0
        total_nox = 0.0
        total_co = 0.0
        total_co2 = 0.0

        fuel_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}

        if not vehicle_records:
            return PollutionEstimate(0, 0, 0, 0, 0, 0, {}, {})

        # Convert duration to fraction of hour
        hour_fraction = duration_seconds / 3600.0

        for record in vehicle_records:
            v_cls = record.get("class_name", "car").lower()
            fuel = record.get("fuel_type", "Unknown")
            speed = max(10.0, record.get("speed_kmh", 40.0))  # Min speed 10 km/h for idling

            fuel_counts[fuel] = fuel_counts.get(fuel, 0) + 1
            category_counts[v_cls] = category_counts.get(v_cls, 0) + 1

            # Get emission factors for vehicle + fuel type
            cls_factors = self.factors.get(v_cls, self.factors.get("car", {}))
            fuel_factors = cls_factors.get(fuel, cls_factors.get("Unknown", {}))

            # Base factors in g/km
            ef_pm25 = fuel_factors.get("PM2_5", 0.015)
            ef_pm10 = fuel_factors.get("PM10", 0.020)
            ef_nox = fuel_factors.get("NOx", 0.400)
            ef_co = fuel_factors.get("CO", 1.000)
            ef_co2 = fuel_factors.get("CO2", 125.0)

            # Speed correction: lower speeds (<20 km/h) increase stop-and-go emissions per km
            speed_mult = 1.0
            if speed < 20.0:
                speed_mult = 1.4
            elif speed > 80.0:
                speed_mult = 1.2

            # Distance traveled in this time slice (km)
            dist_km = speed * hour_fraction

            total_pm25 += ef_pm25 * dist_km * speed_mult
            total_pm10 += ef_pm10 * dist_km * speed_mult
            total_nox += ef_nox * dist_km * speed_mult
            total_co += ef_co * dist_km * speed_mult
            total_co2 += ef_co2 * dist_km * speed_mult

        # Scale to rate in g/hr
        scale_to_hr = 1.0 / max(hour_fraction, 1e-6)

        return PollutionEstimate(
            pm2_5_g_per_hr=round(total_pm25 * scale_to_hr, 4),
            pm10_g_per_hr=round(total_pm10 * scale_to_hr, 4),
            nox_g_per_hr=round(total_nox * scale_to_hr, 4),
            co_g_per_hr=round(total_co * scale_to_hr, 4),
            co2_g_per_hr=round(total_co2 * scale_to_hr, 4),
            total_vehicles=len(vehicle_records),
            fuel_counts=fuel_counts,
            category_counts=category_counts
        )
