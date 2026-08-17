import numpy as np
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Reference urban baseline mass emission rates (grams per hour per km of road section)
# Standard thresholds for normal, moderate, high, and severe urban traffic contributions
BASELINE_EMISSION_THRESHOLDS_G_HR = {
    "PM2.5": 50.0,    # 50 g/hr/km -> Index = 50
    "PM10": 100.0,    # 100 g/hr/km -> Index = 50
    "NO2": 500.0,     # 500 g/hr/km -> Index = 50
    "CO": 2500.0,     # 2500 g/hr/km -> Index = 50
    "SO2": 25.0,      # 25 g/hr/km -> Index = 50
    "CO2": 500000.0   # 500 kg/hr/km -> Index = 50
}

# Sub-index weights for overall composite vehicle pollution contribution index
POLLUTANT_SUBINDEX_WEIGHTS = {
    "NO2": 0.35,
    "PM2.5": 0.30,
    "PM10": 0.15,
    "CO": 0.10,
    "SO2": 0.05,
    "CO2": 0.05
}

class VehiclePollutionIndex:
    """
    Normalized Vehicle Pollution Contribution Index (0 - 100 Scale).
    Transforms estimated vehicle mass emission rates (g/hr) into a standardized urban index.
    
    IMPORTANT SCIENTIFIC DISCLAIMER:
    This score is a relative VEHICLE CONTRIBUTION INDEX derived from traffic flow and EEA emission factors.
    It DOES NOT represent atmospheric ambient pollutant concentration measurements.
    """
    def __init__(self):
        self.disclaimer = (
            "NOTICE: This metric is a normalized VEHICLE POLLUTION CONTRIBUTION INDEX (0-100 scale) "
            "calculated from computer vision traffic counts and EEA COPERT V emission factors. "
            "It does NOT represent direct atmospheric concentration measurements."
        )

    def calculate_sub_index(self, pollutant: str, mass_rate_g_hr: float) -> float:
        """
        Calculates pollutant sub-index (0 to 100 scale).
        """
        baseline = BASELINE_EMISSION_THRESHOLDS_G_HR.get(pollutant, 100.0)
        # Linear scaling up to baseline (Index=50 at baseline emission rate)
        sub_idx = (float(mass_rate_g_hr) / float(baseline)) * 50.0
        # Logarithmic dampening above baseline to prevent runaway spikes
        if sub_idx > 50.0:
            sub_idx = 50.0 + 50.0 * (1.0 - np.exp(-0.02 * (sub_idx - 50.0)))
        return float(np.clip(sub_idx, 0.0, 100.0))

    def compute_composite_index(self, pollutant_emissions_g_hr: Dict[str, float]) -> Dict[str, Any]:
        """
        Computes composite 0-100 Vehicle Pollution Contribution Index and sub-indices.
        """
        sub_indices: Dict[str, float] = {}
        weighted_sum = 0.0

        for pol, weight in POLLUTANT_SUBINDEX_WEIGHTS.items():
            g_hr = pollutant_emissions_g_hr.get(pol, 0.0)
            sub_idx = self.calculate_sub_index(pol, g_hr)
            sub_indices[pol] = round(sub_idx, 2)
            weighted_sum += weight * sub_idx

        composite_idx = float(np.clip(weighted_sum, 0.0, 100.0))

        # Categorize index severity
        if composite_idx < 25.0:
            category = "LOW ESTIMATED VEHICLE CONTRIBUTION"
            color = "#00E676"  # Green
        elif composite_idx < 50.0:
            category = "MODERATE ESTIMATED VEHICLE CONTRIBUTION"
            color = "#FFEA00"  # Yellow
        elif composite_idx < 75.0:
            category = "HIGH ESTIMATED VEHICLE CONTRIBUTION"
            color = "#FF9100"  # Orange
        else:
            category = "SEVERE ESTIMATED VEHICLE CONTRIBUTION"
            color = "#FF1744"  # Red

        return {
            "vehicle_pollution_index": round(composite_idx, 2),
            "category": category,
            "color_code": color,
            "sub_indices": sub_indices,
            "disclaimer": self.disclaimer
        }
