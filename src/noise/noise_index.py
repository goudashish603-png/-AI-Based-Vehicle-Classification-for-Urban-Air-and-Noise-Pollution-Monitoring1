# pyright: reportGeneralTypeIssues=false, reportMissingImports=false, reportUnusedImport=false, reportUnusedVariable=false
# type: ignore
import numpy as np
from typing import Dict, Any

from src.utils import get_logger

# Baseline WHO / EEA urban noise benchmarks (dBA equivalent sound pressure level proxy)
BASELINE_QUIET_DBA = 40.0   # Residential quiet background (Index = 0)
BASELINE_MODERATE_DBA = 60.0 # Urban daylight traffic baseline (Index = 50)
BASELINE_SEVERE_DBA = 85.0   # Heavy industrial/freight traffic limit (Index = 100)

class RelativeNoiseIndex:
    """
    Normalized Relative Noise Pollution Index (0 to 100 Scale).
    Transforms traffic acoustic proxy levels into a standardized 0-100 urban index.
    
    IMPORTANT SCIENTIFIC DISCLAIMER:
    This metric is a RELATIVE NOISE INDEX based on visual traffic volume, vehicle category acoustic weights,
    and CoRTN propagation formulas. It DOES NOT represent calibrated physical microphone measurements.
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.disclaimer = (
            "NOTICE: This metric is a RELATIVE NOISE POLLUTION INDEX (0-100 scale) "
            "estimated from computer vision traffic density and CoRTN acoustic models. "
            "It does NOT represent calibrated physical sound pressure measurements in dB."
        )

    def compute_noise_index(self, estimated_leq_dba: float) -> Dict[str, Any]:
        """
        Transforms estimated L_eq sound level (dBA) into a normalized 0-100 Relative Noise Index.
        """
        self.logger.debug(f"Computing relative noise index for sound proxy: {estimated_leq_dba} dBA")
        leq = float(estimated_leq_dba)
        
        # Scale: 40 dBA -> Index 0, 60 dBA -> Index 50, 85 dBA -> Index 100
        if leq <= BASELINE_QUIET_DBA:
            score = 0.0
        elif leq <= BASELINE_MODERATE_DBA:
            score = ((leq - BASELINE_QUIET_DBA) / (BASELINE_MODERATE_DBA - BASELINE_QUIET_DBA)) * 50.0
        else:
            score = 50.0 + ((leq - BASELINE_MODERATE_DBA) / (BASELINE_SEVERE_DBA - BASELINE_MODERATE_DBA)) * 50.0

        score = float(np.clip(score, 0.0, 100.0))

        # Severity Category
        if score < 25.0:
            category = "LOW RELATIVE TRAFFIC NOISE"
            color = "#00E676"  # Green
        elif score <= 50.0:
            category = "MODERATE RELATIVE TRAFFIC NOISE"
            color = "#FFEA00"  # Yellow
        elif score <= 75.0:
            category = "ELEVATED RELATIVE TRAFFIC NOISE"
            color = "#FF9100"  # Orange
        else:
            category = "HIGH RELATIVE TRAFFIC NOISE"
            color = "#FF1744"  # Red

        return {
            "relative_noise_index": round(score, 2),
            "estimated_leq_dba_proxy": round(leq, 1),
            "category": category,
            "color_code": color,
            "disclaimer": self.disclaimer
        }
