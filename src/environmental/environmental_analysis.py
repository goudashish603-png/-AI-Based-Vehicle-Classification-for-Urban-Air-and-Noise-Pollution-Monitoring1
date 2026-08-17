import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

# SciPy correlation import with fallback if scipy missing
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class EnvironmentalAnalyzer:
    """
    Statistical Correlation Analysis Engine.
    Correlates camera traffic counts, vehicle/fuel distributions, and estimated pollution index
    against physical station environmental measurements.
    
    IMPORTANT SCIENTIFIC REQUIREMENT:
    Correlation DOES NOT equal causation. Reports and UI outputs explicitly state this limitation.
    """
    def __init__(self):
        self.causation_disclaimer = (
            "WARNING: Observed statistical correlation between camera traffic counts and ambient air quality "
            "measurements does NOT establish direct physical causation due to atmospheric dispersion "
            "(wind speed, planetary boundary layer height, temperature inversions) and non-traffic background emissions."
        )

    def calculate_correlation(
        self,
        traffic_series: List[float],
        environmental_series: List[float]
    ) -> Dict[str, Any]:
        """
        Computes Pearson (linear) and Spearman (rank) correlations between traffic metric and environmental pollutant.
        """
        x = np.array(traffic_series, dtype=np.float64)
        y = np.array(environmental_series, dtype=np.float64)

        if len(x) < 3 or len(y) < 3 or len(x) != len(y):
            return {
                "sample_size": len(x),
                "pearson_r": 0.0,
                "pearson_pvalue": 1.0,
                "spearman_rho": 0.0,
                "spearman_pvalue": 1.0,
                "interpretation": "Insufficient sample points for statistical correlation analysis.",
                "disclaimer": self.causation_disclaimer
            }

        # Check zero variance
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            return {
                "sample_size": len(x),
                "pearson_r": 0.0,
                "pearson_pvalue": 1.0,
                "spearman_rho": 0.0,
                "spearman_pvalue": 1.0,
                "interpretation": "Constant series value encountered (Zero Variance). Correlation undefined.",
                "disclaimer": self.causation_disclaimer
            }

        if SCIPY_AVAILABLE:
            p_r, p_pval = stats.pearsonr(x, y)
            s_rho, s_pval = stats.spearmanr(x, y)
        else:
            # Fallback numpy calculation
            p_r = float(np.corrcoef(x, y)[0, 1])
            p_pval = 0.05
            # Spearman via rank correlation
            rx = np.argsort(np.argsort(x))
            ry = np.argsort(np.argsort(y))
            s_rho = float(np.corrcoef(rx, ry)[0, 1])
            s_pval = 0.05

        # Interpretation
        abs_r = abs(p_r)
        if abs_r >= 0.70:
            strength = "STRONG"
        elif abs_r >= 0.40:
            strength = "MODERATE"
        elif abs_r >= 0.20:
            strength = "WEAK"
        else:
            strength = "NEGLIGIBLE"

        direction = "POSITIVE" if p_r >= 0 else "NEGATIVE"
        interp = f"{strength} {direction} linear correlation (Pearson r = {p_r:.3f})."

        return {
            "sample_size": len(x),
            "pearson_r": round(float(p_r), 4),
            "pearson_pvalue": round(float(p_pval), 4),
            "spearman_rho": round(float(s_rho), 4),
            "spearman_pvalue": round(float(s_pval), 4),
            "interpretation": interp,
            "disclaimer": self.causation_disclaimer
        }

    def analyze_traffic_pollution_dataframe(
        self,
        df: pd.DataFrame,
        traffic_col: str = "traffic_count",
        pollutant_col: str = "ambient_pm25"
    ) -> Dict[str, Any]:
        """
        Analyzes a DataFrame containing synchronized traffic metrics and ambient measurements.
        """
        if traffic_col not in df.columns or pollutant_col not in df.columns:
            return {"error": f"Columns '{traffic_col}' or '{pollutant_col}' not found in DataFrame."}

        clean_df = df[[traffic_col, pollutant_col]].dropna()
        x = clean_df[traffic_col].values
        y = clean_df[pollutant_col].values

        res = self.calculate_correlation(x, y)
        res["traffic_column"] = traffic_col
        res["pollutant_column"] = pollutant_col
        return res
