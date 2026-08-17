import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from src.pollution.emission_factors import EmissionFactorRegistry, SUPPORTED_POLLUTANTS
from src.pollution.pollution_index import VehiclePollutionIndex
from src.detection.types import DetectionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AirPollutionEstimator:
    """
    Air Pollution Estimation Engine.
    Calculates estimated vehicle emission contributions based on traffic activity, 
    vehicle types, fuel classifications, and EEA COPERT V emission factors.
    
    IMPORTANT: Estimates emission mass rates (g/hr), NOT atmospheric sensor concentrations.
    """
    def __init__(self, registry: Optional[EmissionFactorRegistry] = None):
        self.registry = registry or EmissionFactorRegistry()
        self.index_calculator = VehiclePollutionIndex()

    def _get_congestion_multiplier(self, speed_kmh: float, is_idling: bool = False) -> float:
        """Determines activity congestion multiplier based on vehicle speed/idling."""
        if is_idling or speed_kmh < 5.0:
            return 2.20  # Stop-and-go heavy congestion idling
        elif speed_kmh < 25.0:
            return 1.50  # Slow urban congestion
        elif speed_kmh < 60.0:
            return 1.00  # Normal urban flow
        else:
            return 1.15  # High speed highway drag

    def estimate_single_vehicle_emissions(
        self,
        vehicle_type: str,
        fuel_type: str,
        distance_km: float = 1.0,
        speed_kmh: float = 30.0,
        is_idling: bool = False
    ) -> Dict[str, float]:
        """
        Calculates estimated mass emissions (in grams) for a single vehicle activity segment.
        Formula: estimated_emission = 1 * emission_factor * (distance * congestion_multiplier)
        """
        v_type = vehicle_type.lower()
        f_type = fuel_type.upper()
        cong_mult = self._get_congestion_multiplier(speed_kmh, is_idling)
        effective_distance = max(0.01, distance_km) * cong_mult

        emissions_g: Dict[str, float] = {}
        for pol in SUPPORTED_POLLUTANTS:
            factor = self.registry.get_factor(v_type, f_type, pol)
            emissions_g[pol] = round(float(factor * effective_distance), 4)

        return emissions_g

    def estimate_fleet_emissions(
        self,
        vehicle_records: List[Dict[str, Any]],
        default_distance_km: float = 1.0,
        time_window_seconds: float = 3600.0
    ) -> Dict[str, Any]:
        """
        Calculates aggregate estimated vehicle emission contributions for a fleet of tracked vehicles.
        
        Args:
            vehicle_records: List of dicts containing 'vehicle_type', 'fuel_type', 'speed_kmh', 'dwell_time_sec'
            default_distance_km: Configurable road segment length proxy (if GPS distance unknown)
            time_window_seconds: Duration of observation window in seconds
        
        Returns:
            Dictionary containing pollutant totals, fuel breakdowns, class breakdowns, and 0-100 Index.
        """
        if not vehicle_records:
            return self._empty_emission_report()

        total_by_pollutant: Dict[str, float] = {pol: 0.0 for pol in SUPPORTED_POLLUTANTS}
        by_fuel_type: Dict[str, Dict[str, float]] = {}
        by_vehicle_type: Dict[str, Dict[str, float]] = {}

        # Scale time window to hourly multiplier
        hourly_scale = 3600.0 / max(1.0, time_window_seconds)

        for rec in vehicle_records:
            v_cls = str(rec.get("vehicle_class", rec.get("vehicle_type", "car"))).lower()
            f_type = str(rec.get("fuel_type", "UNKNOWN")).upper()
            speed = float(rec.get("speed_kmh", 30.0))
            is_idling = bool(rec.get("is_idling", speed < 5.0))
            
            # Use distance proxy if actual distance unknown
            dist_km = float(rec.get("distance_km", default_distance_km))

            single_emissions = self.estimate_single_vehicle_emissions(
                vehicle_type=v_cls,
                fuel_type=f_type,
                distance_km=dist_km,
                speed_kmh=speed,
                is_idling=is_idling
            )

            # Initialize fuel / vehicle class tracking breakdown
            if f_type not in by_fuel_type:
                by_fuel_type[f_type] = {pol: 0.0 for pol in SUPPORTED_POLLUTANTS}
            if v_cls not in by_vehicle_type:
                by_vehicle_type[v_cls] = {pol: 0.0 for pol in SUPPORTED_POLLUTANTS}

            # Accumulate totals
            for pol, val in single_emissions.items():
                total_by_pollutant[pol] += val
                by_fuel_type[f_type][pol] += val
                by_vehicle_type[v_cls][pol] += val

        # Calculate Hourly Mass Emission Rates (g/hr) and Daily Projections (kg/day)
        hourly_emissions_g_hr = {pol: round(total_by_pollutant[pol] * hourly_scale, 2) for pol in SUPPORTED_POLLUTANTS}
        daily_emissions_kg_day = {pol: round((hourly_emissions_g_hr[pol] * 24.0) / 1000.0, 3) for pol in SUPPORTED_POLLUTANTS}

        # Calculate 0-100 Vehicle Pollution Contribution Index
        index_result = self.index_calculator.compute_composite_index(hourly_emissions_g_hr)

        return {
            "observation_window_seconds": time_window_seconds,
            "total_vehicles_analyzed": len(vehicle_records),
            "distance_proxy_km_used": default_distance_km,
            "total_emissions_g": {pol: round(v, 2) for pol, v in total_by_pollutant.items()},
            "hourly_emission_rate_g_hr": hourly_emissions_g_hr,
            "daily_projected_kg_day": daily_emissions_kg_day,
            "by_fuel_type": {f: {pol: round(val, 2) for pol, val in p_dict.items()} for f, p_dict in by_fuel_type.items()},
            "by_vehicle_type": {v: {pol: round(val, 2) for pol, val in p_dict.items()} for v, p_dict in by_vehicle_type.items()},
            "vehicle_pollution_index": index_result
        }

    def _empty_emission_report(self) -> Dict[str, Any]:
        empty_g = {pol: 0.0 for pol in SUPPORTED_POLLUTANTS}
        idx_res = self.index_calculator.compute_composite_index(empty_g)
        return {
            "observation_window_seconds": 0,
            "total_vehicles_analyzed": 0,
            "distance_proxy_km_used": 1.0,
            "total_emissions_g": empty_g,
            "hourly_emission_rate_g_hr": empty_g,
            "daily_projected_kg_day": {pol: 0.0 for pol in SUPPORTED_POLLUTANTS},
            "by_fuel_type": {},
            "by_vehicle_type": {},
            "vehicle_pollution_index": idx_res
        }
