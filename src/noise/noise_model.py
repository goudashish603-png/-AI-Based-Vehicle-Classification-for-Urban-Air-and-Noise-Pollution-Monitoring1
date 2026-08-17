import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from src.noise.noise_index import RelativeNoiseIndex
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Vehicle category acoustic weighting factors (relative noise power multiplier compared to standard car)
VEHICLE_ACOUSTIC_WEIGHTS = {
    "car": 1.0,
    "van": 1.8,
    "truck": 8.0,
    "bus": 6.0,
    "motorcycle": 10.0,
    "ev": 0.2,
    "other_vehicle": 2.0
}

class TrafficNoiseEstimator:
    """
    Video-Based Relative Road Traffic Noise Estimator.
    Calculates equivalent sound level proxy (L_eq dBA) and 0-100 Relative Noise Index
    using CoRTN (Calculation of Road Traffic Noise) acoustic propagation principles.
    
    IMPORTANT METHODOLOGICAL SEPARATION:
    1. Video Relative Noise Estimation: Estimates noise index from vehicle counts & categories.
    2. Audio Classification: Optional spectral sound event classifier.
    3. Calibrated dB Measurements: Requires physical calibrated microphone instrumentation.
    """
    def __init__(
        self,
        base_noise_level_dba: float = 45.0,
        default_distance_meters: float = 10.0
    ):
        self.base_noise_dba = base_noise_level_dba
        self.default_distance_m = max(1.0, default_distance_meters)
        self.index_calculator = RelativeNoiseIndex()

    def calculate_traffic_noise_proxy(
        self,
        vehicle_counts: Dict[str, int],
        avg_speed_kmh: float = 40.0,
        distance_meters: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates estimated traffic sound level proxy (L_eq in dBA) and 0-100 Relative Noise Index.
        
        Args:
            vehicle_counts: Dict mapping vehicle class to active count (e.g. {'car': 5, 'truck': 1})
            avg_speed_kmh: Average traffic flow speed in km/h
            distance_meters: Distance from road centerline in meters (default: 10m)
        """
        dist_m = max(1.0, distance_meters or self.default_distance_m)

        # 1. Calculate Effective Acoustic Volume Q_eff
        q_eff = 0.0
        by_category_contributions: Dict[str, float] = {}

        for cls_name, count in vehicle_counts.items():
            norm_cls = str(cls_name).lower()
            weight = VEHICLE_ACOUSTIC_WEIGHTS.get(norm_cls, 1.5)
            cat_q = count * weight
            q_eff += cat_q
            by_category_contributions[norm_cls] = round(cat_q, 2)

        if q_eff <= 0.0:
            leq_dba = self.base_noise_dba
        else:
            # 2. CoRTN Acoustic Propagation Formula
            # L_eq = Base + 10*log10(Q_eff) + 30*log10(speed/30) - 10*log10(distance/10)
            speed_corr = 30.0 * math.log10(max(10.0, avg_speed_kmh) / 30.0)
            dist_corr = 10.0 * math.log10(dist_m / 10.0)
            
            leq_dba = self.base_noise_dba + 10.0 * math.log10(q_eff) + speed_corr - dist_corr

        leq_dba = round(float(np.clip(leq_dba, 35.0, 95.0)), 1)
        index_res = self.index_calculator.compute_noise_index(leq_dba)

        return {
            "estimated_leq_dba": leq_dba,
            "effective_acoustic_volume_q_eff": round(q_eff, 2),
            "distance_meters_used": dist_m,
            "avg_speed_kmh_used": avg_speed_kmh,
            "category_acoustic_contributions": by_category_contributions,
            "relative_noise_index": index_res
        }
