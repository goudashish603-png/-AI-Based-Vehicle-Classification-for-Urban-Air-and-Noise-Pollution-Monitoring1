from src.pollution.emission_factors import EmissionFactorRegistry, SUPPORTED_POLLUTANTS
from src.pollution.emission_model import AirPollutionEstimator
from src.pollution.pollution_index import VehiclePollutionIndex, BASELINE_EMISSION_THRESHOLDS_G_HR

# Alias for backwards compatibility
AirPollutionEstimator = AirPollutionEstimator

__all__ = [
    "EmissionFactorRegistry",
    "AirPollutionEstimator",
    "VehiclePollutionIndex",
    "SUPPORTED_POLLUTANTS",
    "BASELINE_EMISSION_THRESHOLDS_G_HR"
]
