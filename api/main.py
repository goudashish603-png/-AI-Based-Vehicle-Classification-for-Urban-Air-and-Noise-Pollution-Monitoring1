import cv2
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict, Any

from api.schemas import (
    FuelInferRequest,
    FuelInferResponse,
    EmissionsEstimateRequest,
    EmissionsEstimateResponse,
    ImageInferenceResponse,
    DetectionBox
)

from src.fuel_mapping.fuel_mapper import FuelTypeMapper
from src.pollution.emission_model import AirPollutionEstimator
from src.noise.noise_model import TrafficNoiseEstimator
from src.detection.detector import VehicleDetector
from src.classification.inference import VehicleMakeModelClassifier

app = FastAPI(
    title="AI Vehicle Classification & Environmental Monitoring API",
    description="Production REST API for Real-time Vehicle Detection, Fine-grained Fuel Mapping, Air Pollution ($g/hr$), and Traffic Noise Telemetry.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external web/mobile client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global core component singletons
fuel_mapper = FuelTypeMapper()
pollution_estimator = AirPollutionEstimator()
noise_estimator = TrafficNoiseEstimator()
detector = None
classifier = None

def get_detector() -> VehicleDetector:
    global detector
    if detector is None:
        detector = VehicleDetector(conf_threshold=0.40)
    return detector

def get_classifier() -> VehicleMakeModelClassifier:
    global classifier
    if classifier is None:
        try:
            classifier = VehicleMakeModelClassifier()
        except Exception:
            classifier = None
    return classifier

from fastapi.responses import FileResponse

@app.get("/", tags=["General"])
def root():
    index_path = Path("index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "status": "online",
        "service": "AI-Based Vehicle Classification & Environmental Telemetry API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "detector_status": "ready",
        "fuel_mapper_status": "ready"
    }

@app.post("/api/v1/infer-fuel", response_model=FuelInferResponse, tags=["Fuel Mapping"])
def infer_fuel(req: FuelInferRequest):
    """
    Infers fuel classification using multi-stage database lookup and fleet prior distribution fallback.
    """
    try:
        res = fuel_mapper.infer_fuel_type(
            manufacturer=req.manufacturer,
            model=req.model,
            variant=req.variant or "",
            vehicle_type=req.vehicle_type or "",
            track_id=req.track_id or 0
        )
        return FuelInferResponse(
            fuel_type=res.fuel_type,
            confidence=res.confidence,
            mapping_source=res.mapping_source,
            mapping_notes=res.mapping_notes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fuel inference error: {str(e)}")

@app.post("/api/v1/estimate-emissions", response_model=EmissionsEstimateResponse, tags=["Environmental Telemetry"])
def estimate_emissions(req: EmissionsEstimateRequest):
    """
    Calculates estimated air pollution mass rates ($g/hr$) and traffic acoustic noise proxies from vehicle category counts.
    """
    try:
        records = []
        for cat, cnt in req.vehicle_counts.items():
            for _ in range(cnt):
                records.append({"vehicle_type": cat.lower(), "fuel_type": "PETROL"})

        fleet_pollution = pollution_estimator.estimate_fleet_emissions(records)
        fleet_noise = noise_estimator.calculate_traffic_noise_proxy(req.vehicle_counts)

        p_info = fleet_pollution.get("vehicle_pollution_index", {})
        if isinstance(p_info, dict):
            p_idx = float(p_info.get("vehicle_pollution_index", 0.0))
            p_cat = str(p_info.get("category", "MODERATE ESTIMATED VEHICLE CONTRIBUTION"))
        else:
            p_idx = float(p_info)
            p_cat = "MODERATE ESTIMATED VEHICLE CONTRIBUTION"

        n_info = fleet_noise.get("relative_noise_index", {})
        if isinstance(n_info, dict):
            n_idx = float(n_info.get("relative_noise_index", 0.0))
            n_cat = str(n_info.get("category", "MODERATE RELATIVE TRAFFIC NOISE"))
        else:
            n_idx = float(n_info)
            n_cat = "MODERATE RELATIVE TRAFFIC NOISE"

        return EmissionsEstimateResponse(
            vehicle_pollution_index=p_idx,
            relative_noise_index=n_idx,
            hourly_emissions_g_hr=fleet_pollution.get("hourly_emission_rate_g_hr", {}),
            category_pollution=p_cat,
            category_noise=n_cat
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emissions estimation error: {str(e)}")

@app.post("/api/v1/predict-image", response_model=ImageInferenceResponse, tags=["Computer Vision Pipeline"])
async def predict_image(file: UploadFile = File(...)):
    """
    Processes an uploaded traffic image through YOLO detection, classifier, fuel mapper, and environmental engines.
    """
    t0 = time.time()
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image file.")

        det_engine = get_detector()
        cls_engine = get_classifier()

        # Resize image for fast inference if width > 1280
        h, w = img.shape[:2]
        if w > 1280:
            scale = 1280.0 / w
            img = cv2.resize(img, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)

        detections = det_engine.detect(img)

        detection_boxes = []
        vehicle_records = []
        fuel_counts: Dict[str, int] = {}

        for det in detections:
            raw_fuel = "Petrol"
            raw_conf = float(det.confidence)
            probs = None

            if cls_engine and det.vehicle_crop is not None:
                try:
                    raw_fuel, raw_conf, probs = cls_engine.predict_crop(det.vehicle_crop, det.class_name)
                except Exception:
                    pass

            final_fuel, final_conf = fuel_mapper.map_fuel_type(det.class_name, raw_fuel, raw_conf, probs)
            det.fuel_type = final_fuel
            det.fuel_confidence = final_conf

            fuel_counts[final_fuel] = fuel_counts.get(final_fuel, 0) + 1
            vehicle_records.append({
                "vehicle_type": det.class_name.lower(),
                "fuel_type": final_fuel,
                "speed_kmh": 40.0
            })

            x1, y1, x2, y2 = det.bbox.to_int_tuple()
            detection_boxes.append(DetectionBox(
                track_id=det.track_id,
                class_name=det.class_name,
                confidence=round(float(det.confidence), 4),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                fuel_type=final_fuel,
                fuel_confidence=round(float(final_conf), 4)
            ))

        fleet_pollution = pollution_estimator.estimate_fleet_emissions(vehicle_records)
        fleet_noise = noise_estimator.calculate_traffic_noise_proxy(
            {v["vehicle_type"]: 1 for v in vehicle_records}
        )

        p_info = fleet_pollution.get("vehicle_pollution_index", {})
        if isinstance(p_info, dict):
            p_idx = float(p_info.get("vehicle_pollution_index", 0.0))
        else:
            p_idx = float(p_info)

        n_info = fleet_noise.get("relative_noise_index", {})
        if isinstance(n_info, dict):
            n_idx = float(n_info.get("relative_noise_index", 0.0))
        else:
            n_idx = float(n_info)

        proc_ms = round((time.time() - t0) * 1000.0, 2)

        return ImageInferenceResponse(
            vehicle_count=len(detections),
            fuel_counts=fuel_counts,
            pollution_index=p_idx,
            relative_noise_index=n_idx,
            pollutants_g_hr=fleet_pollution.get("hourly_emission_rate_g_hr", {}),
            detections=detection_boxes,
            processing_time_ms=proc_ms
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inference pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
