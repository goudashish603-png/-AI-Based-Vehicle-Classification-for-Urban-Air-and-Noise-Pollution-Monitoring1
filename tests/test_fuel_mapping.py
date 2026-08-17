import pytest
import pandas as pd
from pathlib import Path

from src.fuel_mapping.fuel_mapper import (
    FuelTypeMapper,
    FuelInferenceResult,
    VehicleFuelPrediction,
    ALLOWED_FUEL_TYPES,
    normalize_string
)

def test_exact_matches():
    mapper = FuelTypeMapper()

    # Toyota Prius -> HYBRID
    res_prius = mapper.infer_fuel_type("Toyota", "Prius")
    assert res_prius.fuel_type == "HYBRID"
    assert res_prius.confidence >= 0.90
    assert "Exact" in res_prius.mapping_source

    # Tesla Model 3 -> EV
    res_tesla = mapper.infer_fuel_type("Tesla", "Model 3")
    assert res_tesla.fuel_type == "EV"
    assert res_tesla.confidence == 1.0

def test_normalized_matching():
    mapper = FuelTypeMapper()
    # Spacing/case variations
    res = mapper.infer_fuel_type("  tEsLa ", " model 3 ")
    assert res.fuel_type == "EV"
    assert res.confidence > 0.85

def test_fuzzy_matching():
    mapper = FuelTypeMapper()
    # Slight typo "Teslla"
    res = mapper.infer_fuel_type("Teslla", "Model 3")
    assert res.fuel_type == "EV"
    assert "Fuzzy" in res.mapping_source

def test_ambiguous_model_handling():
    mapper = FuelTypeMapper()
    # BMW 3 Series exists in Petrol, Diesel, Hybrid
    res = mapper.infer_fuel_type("BMW", "3 Series")
    assert res.fuel_type == "AMBIGUOUS"
    assert res.confidence == 0.50
    assert "Ambiguous" in res.mapping_source

def test_unknown_model_handling():
    mapper = FuelTypeMapper()
    res = mapper.infer_fuel_type("NonExistentBrand", "UnheardModel999")
    assert res.fuel_type == "UNKNOWN"
    assert res.confidence == 0.0

def test_predict_vehicle_fuel_object():
    mapper = FuelTypeMapper()
    pred = mapper.predict_vehicle_fuel(
        vehicle_id="TRK-101",
        manufacturer="Toyota",
        model="Prius"
    )
    assert isinstance(pred, VehicleFuelPrediction)
    assert pred.vehicle_id == "TRK-101"
    assert pred.manufacturer == "Toyota"
    assert pred.model == "Prius"
    assert pred.fuel_type == "HYBRID"
    assert pred.fuel_confidence >= 0.90

def test_admin_update(tmp_path):
    csv_file = tmp_path / "test_mapping.csv"
    with open(csv_file, "w") as f:
        f.write("manufacturer,model,variant,fuel_type,confidence,source,source_url,notes\n")
        f.write("Rivian,R1T,Base,EV,1.0,Official,,Electric truck\n")

    mapper = FuelTypeMapper(csv_path=csv_file)
    res_before = mapper.infer_fuel_type("Rivian", "R1T")
    assert res_before.fuel_type == "EV"

    # Query CLI for Tesla Model 3
    import subprocess, sys
    cmd = [
        sys.executable, "scripts/update_fuel_mapping.py",
        "--query", "Tesla", "Model 3"
    ]
    sub_res = subprocess.run(cmd, capture_output=True, text=True)
    assert "EV" in sub_res.stdout
