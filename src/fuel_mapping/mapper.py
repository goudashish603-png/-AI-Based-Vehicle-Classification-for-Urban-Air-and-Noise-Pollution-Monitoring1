from typing import Dict, Tuple, Optional
import numpy as np

# Regional Prior Distributions P(Fuel | Vehicle Class)
# Based on CPCB / European Environment Agency (EEA) fleet composition statistics
FLEET_PRIORS = {
    "car": {
        "Petrol": 0.48,
        "Diesel": 0.32,
        "Electric Vehicle (EV)": 0.08,
        "CNG/LPG": 0.07,
        "Hybrid": 0.04,
        "Unknown": 0.01
    },
    "bus": {
        "Diesel": 0.55,
        "CNG/LPG": 0.35,
        "Electric Vehicle (EV)": 0.08,
        "Petrol": 0.00,
        "Hybrid": 0.01,
        "Unknown": 0.01
    },
    "truck": {
        "Diesel": 0.88,
        "Petrol": 0.08,
        "CNG/LPG": 0.02,
        "Electric Vehicle (EV)": 0.01,
        "Hybrid": 0.00,
        "Unknown": 0.01
    },
    "motorcycle": {
        "Petrol": 0.90,
        "Electric Vehicle (EV)": 0.08,
        "Diesel": 0.00,
        "CNG/LPG": 0.00,
        "Hybrid": 0.01,
        "Unknown": 0.01
    }
}

class FuelTypeMapper:
    """
    Probabilistic Vehicle-to-Fuel Type Mapper.
    Combines fine-grained visual classifier logits with regional fleet priors (Bayesian fusion).
    """
    def __init__(self, prior_weight: float = 0.20):
        self.prior_weight = prior_weight

    def map_fuel_type(
        self,
        vehicle_class: str,
        visual_pred_fuel: str,
        visual_confidence: float,
        visual_probs: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float]:
        """
        Calculates posterior probability distribution over fuel types.
        
        Args:
            vehicle_class: e.g. 'car', 'bus', 'truck', 'motorcycle'
            visual_pred_fuel: Predicted fuel type from fine-grained classifier
            visual_confidence: Confidence of visual prediction (0.0 to 1.0)
            visual_probs: Dictionary of all class probabilities from classifier
            
        Returns:
            Tuple of (final_predicted_fuel_type, final_confidence)
        """
        v_cls = vehicle_class.lower()
        priors = FLEET_PRIORS.get(v_cls, FLEET_PRIORS["car"])

        if visual_probs is None or visual_confidence < 0.25:
            # If visual confidence is very low, rely on fleet prior
            top_fuel = max(priors, key=priors.get)
            return top_fuel, float(priors[top_fuel])

        # Bayesian fusion: P(Fuel | Vision & Class) ~ Vision_Prob^(1-w) * Prior^w
        posterior = {}
        total = 0.0
        for fuel_type, prior_p in priors.items():
            vis_p = visual_probs.get(fuel_type, 0.01)
            # Weighted log-linear combination
            comb = (vis_p ** (1.0 - self.prior_weight)) * ((prior_p + 1e-4) ** self.prior_weight)
            posterior[fuel_type] = comb
            total += comb

        # Normalize
        for k in posterior:
            posterior[k] /= max(total, 1e-6)

        final_fuel = max(posterior, key=posterior.get)
        final_conf = float(posterior[final_fuel])

        return final_fuel, final_conf
