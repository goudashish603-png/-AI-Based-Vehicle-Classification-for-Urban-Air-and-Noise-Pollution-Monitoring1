# type: ignore
import cv2
import time
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Any

from src.detection.detector import VehicleDetector
from src.tracking.tracker import VehicleTracker
from src.classification.inference import VehicleMakeModelClassifier
from src.fuel_mapping.fuel_mapper import FuelTypeMapper
from src.pollution.emission_model import AirPollutionEstimator
from src.noise.noise_model import TrafficNoiseEstimator
from src.utils.device import get_device
from src.utils.profiler import PerformanceProfiler
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EndToEndPipeline:
    """
    Real-Time Optimized End-to-End Inference Pipeline.
    
    Optimizations:
    - Frame Skipping (`process_every_n_frames`): Runs YOLO detection every N frames, reusing object tracks on intermediate frames.
    - Batch Crop Classification (`predict_batch`): Batches PyTorch classifier crops in single tensor passes.
    - Performance Profiling: Measures latency breakdown (detection, tracking, classification) and process RAM memory.
    """
    def __init__(
        self,
        yolo_weights: str = "yolov8n.pt",
        classifier_weights: Optional[str] = None,
        conf_threshold: float = 0.40,
        device_pref: str = "auto",
        process_every_n_frames: int = 2
    ):
        self.device = get_device(device_pref)
        self.conf_threshold = conf_threshold
        self.process_every_n_frames = max(1, process_every_n_frames)

        logger.info(f"Initializing Real-Time Optimized Pipeline (device={self.device}, process_every_n_frames={self.process_every_n_frames})...")
        
        # 1. Detection Engine
        self.detector = VehicleDetector(
            model_path=yolo_weights,
            conf_threshold=conf_threshold
        )

        # 2. Tracking Engine
        self.tracker = VehicleTracker(fps=25.0)

        # 3. Make/Model Classifier (Optional Fallback)
        self.classifier = None
        try:
            self.classifier = VehicleMakeModelClassifier(
                weights_path=classifier_weights,
                device_pref=device_pref
            )
            logger.info("Vehicle Make/Model Classifier loaded successfully.")
        except Exception as e:
            logger.warning(f"Classifier loading skipped ({e}). Operating with detector vehicle classes.")

        # 4. Fuel-Type Mapper
        self.fuel_mapper = FuelTypeMapper()

        # 5. Air & Noise Pollution Estimators
        self.pollution_estimator = AirPollutionEstimator()
        self.noise_estimator = TrafficNoiseEstimator()

        # 6. Performance Profiler
        self.profiler = PerformanceProfiler()

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_dir: Union[str, Path] = "outputs/predictions"
    ) -> Dict[str, Any]:
        """
        Processes a single input image through the pipeline with timing profiling.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.profiler.reset()

        if isinstance(image_input, (str, Path)):
            img_p = Path(image_input)
            if not img_p.exists():
                raise FileNotFoundError(f"Input image not found: {img_p}")
            frame = cv2.imread(str(img_p))
        else:
            frame = image_input.copy()

        if frame is None:
            raise ValueError("Failed to decode input image.")

        # Preprocessing
        # Detection
        t_det_start = time.time()
        detections = self.detector.detect(frame, conf_thresh=self.conf_threshold)
        self.profiler.detection_ms += (time.time() - t_det_start) * 1000.0

        # Tracking
        t_trk_start = time.time()
        tracked_dets = self.tracker.update(detections, frame_idx=0)
        self.profiler.tracking_ms += (time.time() - t_trk_start) * 1000.0

        # Batch Crop Extraction
        crops = []
        rec_templates = []
        fuel_counts: Dict[str, int] = {"PETROL": 0, "DIESEL": 0, "EV": 0, "CNG_LPG": 0, "HYBRID": 0, "UNKNOWN": 0, "AMBIGUOUS": 0}
        type_counts: Dict[str, int] = {}

        for det in tracked_dets:
            tid = det.track_id or 1
            v_type = det.class_name.lower()
            v_conf = float(det.confidence)

            x1, y1, x2, y2 = det.bbox.to_int_tuple()
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

            type_counts[v_type] = type_counts.get(v_type, 0) + 1

            if crop.size > 0:
                crops.append(crop)
                rec_templates.append((tid, v_type, v_conf))

        # Batch Make/Model Inference
        t_cls_start = time.time()
        classified_results = []
        if self.classifier and len(crops) > 0:
            try:
                classified_results = self.classifier.predict_batch(crops)
            except Exception:
                classified_results = [("Generic", v_t.capitalize(), 0.50, []) for _, v_t, _ in rec_templates]
        else:
            classified_results = [("Generic", v_t.capitalize(), 0.50, []) for _, v_t, _ in rec_templates]
        self.profiler.classification_ms += (time.time() - t_cls_start) * 1000.0

        vehicle_records = []
        for (tid, v_type, v_conf), (mfr, model, m_conf, _) in zip(rec_templates, classified_results):
            fuel_res = self.fuel_mapper.infer_fuel_type(mfr, model, vehicle_type=v_type, track_id=tid)
            f_type = fuel_res.fuel_type
            f_conf = fuel_res.confidence

            fuel_counts[f_type] = fuel_counts.get(f_type, 0) + 1

            sing_em = self.pollution_estimator.estimate_single_vehicle_emissions(v_type, f_type)
            est_p_score = round(sing_em.get("PM2.5", 0.0) + sing_em.get("NO2", 0.0), 3)

            sing_n = self.noise_estimator.calculate_traffic_noise_proxy({v_type: 1})
            n_score = sing_n["estimated_leq_dba"]

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            rec = {
                "track_id": tid,
                "vehicle_type": v_type,
                "vehicle_confidence": round(v_conf, 4),
                "manufacturer": mfr,
                "model": model,
                "model_confidence": round(m_conf, 4),
                "fuel_type": f_type,
                "fuel_confidence": round(f_conf, 4),
                "estimated_pollution_score": est_p_score,
                "noise_score": n_score,
                "timestamp": ts,
                "speed_kmh": 30.0
            }
            vehicle_records.append(rec)

        # Fleet Level Pollution & Noise Aggregations
        fleet_pollution = self.pollution_estimator.estimate_fleet_emissions(vehicle_records)
        fleet_noise = self.noise_estimator.calculate_traffic_noise_proxy(type_counts)

        perf_summary = self.profiler.get_performance_summary(frame_count=1)

        # Build vehicles.csv DataFrame
        cols = [
            "track_id", "vehicle_type", "vehicle_confidence", "manufacturer",
            "model", "model_confidence", "fuel_type", "fuel_confidence",
            "estimated_pollution_score", "noise_score", "timestamp"
        ]
        vehicles_df = pd.DataFrame(vehicle_records, columns=cols)
        csv_path = out_dir / "vehicles.csv"
        vehicles_df.to_csv(csv_path, index=False)

        # Build summary.json
        summary = {
            "total_unique_vehicles": len(vehicle_records),
            "petrol_count": fuel_counts.get("PETROL", 0),
            "diesel_count": fuel_counts.get("DIESEL", 0),
            "ev_count": fuel_counts.get("EV", 0),
            "cng_lpg_count": fuel_counts.get("CNG_LPG", 0),
            "hybrid_count": fuel_counts.get("HYBRID", 0),
            "unknown_count": fuel_counts.get("UNKNOWN", 0) + fuel_counts.get("AMBIGUOUS", 0),
            "vehicle_type_counts": type_counts,
            "pollution_index": fleet_pollution["vehicle_pollution_index"],
            "noise_index": fleet_noise["relative_noise_index"],
            "pollutant_estimates": fleet_pollution["hourly_emission_rate_g_hr"],
            "processing_fps": perf_summary["processing_fps"],
            "performance_telemetry": perf_summary
        }

        json_path = out_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Render Annotated Output Image
        annotated = self.tracker.draw_tracking_overlay(frame)
        cv2.imwrite(str(out_dir / "annotated_image.jpg"), annotated)

        logger.info(f"Processed image at {perf_summary['processing_fps']} FPS (RAM: {perf_summary['memory_usage_mb']} MB). Saved to {out_dir}")
        return summary

    def process_video(
        self,
        video_input: Union[str, Path],
        output_dir: Union[str, Path] = "outputs/predictions",
        save_video: bool = True,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Processes an input video file with frame-skipping, max_frames cap, and profiling.
        """
        v_path = Path(video_input)
        if not v_path.exists():
            raise FileNotFoundError(f"Input video file not found: {v_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {v_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        total_frames = min(raw_total_frames, max_frames) if (max_frames and raw_total_frames > 0) else (max_frames or raw_total_frames)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.tracker.fps = fps_in

        writer = None
        if save_video:
            try:
                out_vid = out_dir / "tracked_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_vid), fourcc, fps_in, (width, height))
            except Exception as vid_err:
                logger.warning(f"VideoWriter initialization skipped ({vid_err}). Output video saving disabled.")
                writer = None

        self.profiler.reset()
        frame_idx = 0

        fuel_counts: Dict[str, int] = {"PETROL": 0, "DIESEL": 0, "EV": 0, "CNG_LPG": 0, "HYBRID": 0, "UNKNOWN": 0, "AMBIGUOUS": 0}
        type_counts: Dict[str, int] = {}
        tracked_vehicles_map: Dict[int, Dict[str, Any]] = {}
        last_detections = []

        logger.info(f"Processing video {v_path} (frame_step={self.process_every_n_frames}, max_frames={max_frames})...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames is not None and frame_idx >= max_frames):
                break

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Frame Skipping Optimization: run detection every N frames
            if frame_idx % self.process_every_n_frames == 0:
                t_det = time.time()
                detections = self.detector.detect(frame, conf_thresh=self.conf_threshold)
                self.profiler.detection_ms += (time.time() - t_det) * 1000.0
                last_detections = detections
            else:
                detections = last_detections

            # Multi-Object Tracking
            t_trk = time.time()
            tracked_dets = self.tracker.update(detections, frame_idx=frame_idx, timestamp=ts)
            self.profiler.tracking_ms += (time.time() - t_trk) * 1000.0

            # Batch Crop Extraction for New Tracks
            crops = []
            new_track_rec = []
            for det in tracked_dets:
                tid = det.track_id
                v_type = det.class_name.lower()
                v_conf = float(det.confidence)

                if tid not in tracked_vehicles_map:
                    x1, y1, x2, y2 = det.bbox.to_int_tuple()
                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    if crop.size > 0:
                        crops.append(crop)
                        new_track_rec.append((tid, v_type, v_conf))

            # Batch Classification
            t_cls = time.time()
            classified_results = []
            if self.classifier and len(crops) > 0:
                try:
                    classified_results = self.classifier.predict_batch(crops)
                except Exception:
                    classified_results = [("Generic", vt.capitalize(), 0.50, []) for _, vt, _ in new_track_rec]
            else:
                classified_results = [("Generic", vt.capitalize(), 0.50, []) for _, vt, _ in new_track_rec]
            self.profiler.classification_ms += (time.time() - t_cls) * 1000.0

            for (tid, v_type, v_conf), (mfr, model, m_conf, _) in zip(new_track_rec, classified_results):
                fuel_res = self.fuel_mapper.infer_fuel_type(mfr, model, vehicle_type=v_type, track_id=tid or 0)
                f_type = fuel_res.fuel_type
                f_conf = fuel_res.confidence

                sing_em = self.pollution_estimator.estimate_single_vehicle_emissions(v_type, f_type)
                est_p_score = round(sing_em.get("PM2.5", 0.0) + sing_em.get("NO2", 0.0), 3)

                sing_n = self.noise_estimator.calculate_traffic_noise_proxy({v_type: 1})
                n_score = sing_n["estimated_leq_dba"]

                rec = {
                    "track_id": tid,
                    "vehicle_type": v_type,
                    "vehicle_confidence": round(v_conf, 4),
                    "manufacturer": mfr,
                    "model": model,
                    "model_confidence": round(m_conf, 4),
                    "fuel_type": f_type,
                    "fuel_confidence": round(f_conf, 4),
                    "estimated_pollution_score": est_p_score,
                    "noise_score": n_score,
                    "timestamp": ts,
                    "speed_kmh": 30.0
                }
                tracked_vehicles_map[tid] = rec

                fuel_counts[f_type] = fuel_counts.get(f_type, 0) + 1
                type_counts[v_type] = type_counts.get(v_type, 0) + 1

            if writer:
                annotated = self.tracker.draw_tracking_overlay(frame)
                writer.write(annotated)

            frame_idx += 1

            if progress_callback and (frame_idx % 5 == 0 or frame_idx == 1):
                try:
                    progress_callback(frame_idx, total_frames)
                except Exception:
                    pass

        cap.release()
        if writer:
            writer.release()

        perf_summary = self.profiler.get_performance_summary(frame_count=frame_idx)
        all_vehicle_records = list(tracked_vehicles_map.values())

        fleet_pollution = self.pollution_estimator.estimate_fleet_emissions(
            all_vehicle_records,
            time_window_seconds=max(1.0, frame_idx / fps_in)
        )
        fleet_noise = self.noise_estimator.calculate_traffic_noise_proxy(type_counts)

        # Export vehicles.csv
        cols = [
            "track_id", "vehicle_type", "vehicle_confidence", "manufacturer",
            "model", "model_confidence", "fuel_type", "fuel_confidence",
            "estimated_pollution_score", "noise_score", "timestamp"
        ]
        vehicles_df = pd.DataFrame(all_vehicle_records, columns=cols)
        csv_path = out_dir / "vehicles.csv"
        vehicles_df.to_csv(csv_path, index=False)

        # Export summary.json
        summary = {
            "total_unique_vehicles": len(all_vehicle_records),
            "petrol_count": fuel_counts.get("PETROL", 0),
            "diesel_count": fuel_counts.get("DIESEL", 0),
            "ev_count": fuel_counts.get("EV", 0),
            "cng_lpg_count": fuel_counts.get("CNG_LPG", 0),
            "hybrid_count": fuel_counts.get("HYBRID", 0),
            "unknown_count": fuel_counts.get("UNKNOWN", 0) + fuel_counts.get("AMBIGUOUS", 0),
            "vehicle_type_counts": type_counts,
            "pollution_index": fleet_pollution["vehicle_pollution_index"],
            "noise_index": fleet_noise["relative_noise_index"],
            "pollutant_estimates": fleet_pollution["hourly_emission_rate_g_hr"],
            "processing_fps": perf_summary["processing_fps"],
            "performance_telemetry": perf_summary
        }

        json_path = out_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Processed {frame_idx} video frames at {perf_summary['processing_fps']} FPS (RAM: {perf_summary['memory_usage_mb']} MB). Saved to {out_dir}")
        return summary
