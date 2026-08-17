from src.noise.noise_model import TrafficNoiseEstimator, VEHICLE_ACOUSTIC_WEIGHTS
from src.noise.noise_index import RelativeNoiseIndex
from src.noise.audio_classifier import AudioNoiseClassifier, AUDIO_CLASSES

# Backwards compatibility alias
NoisePollutionEstimator = TrafficNoiseEstimator

__all__ = [
    "TrafficNoiseEstimator",
    "NoisePollutionEstimator",
    "RelativeNoiseIndex",
    "AudioNoiseClassifier",
    "VEHICLE_ACOUSTIC_WEIGHTS",
    "AUDIO_CLASSES"
]
