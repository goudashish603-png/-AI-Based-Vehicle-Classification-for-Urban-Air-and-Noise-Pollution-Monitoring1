import pytest
import numpy as np

from src.noise.noise_model import TrafficNoiseEstimator, VEHICLE_ACOUSTIC_WEIGHTS
from src.noise.noise_index import RelativeNoiseIndex
from src.noise.audio_classifier import AudioNoiseClassifier, AUDIO_CLASSES

def test_traffic_noise_estimator_weights():
    estimator = TrafficNoiseEstimator()
    
    # 5 Cars vs 5 Trucks
    res_cars = estimator.calculate_traffic_noise_proxy({"car": 5})
    res_trucks = estimator.calculate_traffic_noise_proxy({"truck": 5})
    
    assert res_trucks["estimated_leq_dba"] > res_cars["estimated_leq_dba"]
    assert res_trucks["effective_acoustic_volume_q_eff"] > res_cars["effective_acoustic_volume_q_eff"]

def test_ev_low_acoustic_footprint():
    estimator = TrafficNoiseEstimator()
    res_ev = estimator.calculate_traffic_noise_proxy({"ev": 10})
    res_moto = estimator.calculate_traffic_noise_proxy({"motorcycle": 10})
    
    # EV 10 count must be quieter than Motorcycle 10 count
    assert res_ev["estimated_leq_dba"] < res_moto["estimated_leq_dba"]

def test_distance_propagation():
    estimator = TrafficNoiseEstimator()
    # Same traffic at 10m vs 50m
    res_near = estimator.calculate_traffic_noise_proxy({"car": 10}, distance_meters=10.0)
    res_far = estimator.calculate_traffic_noise_proxy({"car": 10}, distance_meters=50.0)
    
    assert res_far["estimated_leq_dba"] < res_near["estimated_leq_dba"]

def test_relative_noise_index():
    index_calc = RelativeNoiseIndex()
    res = index_calc.compute_noise_index(estimated_leq_dba=60.0)
    
    assert res["relative_noise_index"] == 50.0
    assert "MODERATE" in res["category"]
    assert "RELATIVE" in res["disclaimer"]

def test_audio_noise_classifier():
    classifier = AudioNoiseClassifier()
    dummy_wav = np.random.normal(0, 0.1, 32000).astype(np.float32)
    
    res = classifier.classify_audio(dummy_wav)
    assert res["predicted_class"] in AUDIO_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert "disclaimer" in res
