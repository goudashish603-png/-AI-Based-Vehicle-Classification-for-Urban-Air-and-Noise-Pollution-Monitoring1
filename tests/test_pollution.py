import pytest
from src.pollution.emission_factors import EmissionFactorRegistry, SUPPORTED_POLLUTANTS
from src.pollution.emission_model import AirPollutionEstimator
from src.pollution.pollution_index import VehiclePollutionIndex

def test_emission_factor_registry():
    registry = EmissionFactorRegistry()
    
    # Petrol Car PM2.5 factor
    p_factor = registry.get_factor("car", "PETROL", "PM2.5")
    assert p_factor == 0.005

    # Diesel Truck NO2 factor (higher than petrol car)
    d_truck_no2 = registry.get_factor("truck", "DIESEL", "NO2")
    assert d_truck_no2 > p_factor

    # EV Car tailpipe NO2 factor must be ZERO
    ev_no2 = registry.get_factor("car", "EV", "NO2")
    assert ev_no2 == 0.0

def test_single_vehicle_emission_formula():
    estimator = AirPollutionEstimator()
    # 1 km distance, 30 km/h speed
    res_petrol = estimator.estimate_single_vehicle_emissions("car", "PETROL", distance_km=1.0, speed_kmh=30.0)
    assert res_petrol["PM2.5"] == 0.005
    assert res_petrol["CO2"] == 145.0

    # Idling congestion multiplier (2.2x)
    res_idling = estimator.estimate_single_vehicle_emissions("car", "PETROL", distance_km=1.0, speed_kmh=0.0, is_idling=True)
    assert pytest.approx(res_idling["CO2"], 0.01) == 145.0 * 2.20

def test_fleet_emissions_and_breakdowns():
    estimator = AirPollutionEstimator()
    fleet = [
        {"vehicle_class": "car", "fuel_type": "PETROL", "speed_kmh": 30.0},
        {"vehicle_class": "car", "fuel_type": "EV", "speed_kmh": 30.0},
        {"vehicle_class": "truck", "fuel_type": "DIESEL", "speed_kmh": 20.0}
    ]

    report = estimator.estimate_fleet_emissions(fleet, default_distance_km=1.0, time_window_seconds=3600.0)
    
    assert report["total_vehicles_analyzed"] == 3
    assert "PETROL" in report["by_fuel_type"]
    assert "EV" in report["by_fuel_type"]
    assert "DIESEL" in report["by_fuel_type"]
    assert "car" in report["by_vehicle_type"]
    assert "truck" in report["by_vehicle_type"]
    
    # EV tailpipe NO2 must be 0
    assert report["by_fuel_type"]["EV"]["NO2"] == 0.0

def test_vehicle_pollution_index_math():
    index_calc = VehiclePollutionIndex()
    
    # Baseline emissions -> composite index around 50
    baseline_rates = {
        "PM2.5": 50.0,
        "PM10": 100.0,
        "NO2": 500.0,
        "CO": 2500.0,
        "SO2": 25.0,
        "CO2": 500000.0
    }
    
    res = index_calc.compute_composite_index(baseline_rates)
    assert 0.0 <= res["vehicle_pollution_index"] <= 100.0
    assert pytest.approx(res["vehicle_pollution_index"], 0.1) == 50.0
    assert "INDEX" in res["disclaimer"]
