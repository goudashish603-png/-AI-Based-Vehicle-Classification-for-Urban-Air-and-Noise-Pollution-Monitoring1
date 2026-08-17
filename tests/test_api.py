import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from api.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_infer_fuel_endpoint():
    payload = {
        "manufacturer": "Toyota",
        "model": "Prius",
        "variant": "",
        "vehicle_type": "car",
        "track_id": 101
    }
    response = client.post("/api/v1/infer-fuel", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fuel_type"] == "HYBRID"
    assert data["confidence"] >= 0.80

def test_estimate_emissions_endpoint():
    payload = {
        "vehicle_counts": {"car": 5, "truck": 2, "bus": 1},
        "duration_seconds": 3600.0
    }
    response = client.post("/api/v1/estimate-emissions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "vehicle_pollution_index" in data
    assert "relative_noise_index" in data
    assert "hourly_emissions_g_hr" in data

def test_predict_image_endpoint():
    sample_path = Path("data/raw/images/traffic_sample_1.jpg")
    if not sample_path.exists():
        pytest.skip("Sample image not found.")

    with open(sample_path, "rb") as f:
        files = {"file": ("traffic_sample_1.jpg", f, "image/jpeg")}
        response = client.post("/api/v1/predict-image", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "vehicle_count" in data
    assert "fuel_counts" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)
