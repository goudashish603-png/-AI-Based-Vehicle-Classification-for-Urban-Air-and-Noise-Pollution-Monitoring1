import os
import json
import pytest
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

from src.pipeline import EndToEndPipeline

def test_pipeline_image_processing(tmp_path):
    # Create dummy synthetic traffic image
    img_p = tmp_path / "test_image.jpg"
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 120
    # Draw simple vehicle shape
    cv2.rectangle(dummy_img, (100, 100), (300, 250), (50, 50, 200), -1)
    cv2.imwrite(str(img_p), dummy_img)

    pipeline = EndToEndPipeline(conf_threshold=0.10)
    out_dir = tmp_path / "predictions"
    summary = pipeline.process_image(img_p, output_dir=out_dir)

    # Check output files exist
    csv_file = out_dir / "vehicles.csv"
    json_file = out_dir / "summary.json"
    assert csv_file.exists()
    assert json_file.exists()

    # Verify CSV schema
    df = pd.read_csv(csv_file)
    expected_cols = [
        "track_id", "vehicle_type", "vehicle_confidence", "manufacturer",
        "model", "model_confidence", "fuel_type", "fuel_confidence",
        "estimated_pollution_score", "noise_score", "timestamp"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify JSON summary structure
    with open(json_file) as f:
        summary_data = json.load(f)

    expected_json_keys = [
        "total_unique_vehicles", "petrol_count", "diesel_count", "ev_count",
        "cng_lpg_count", "hybrid_count", "unknown_count", "vehicle_type_counts",
        "pollution_index", "noise_index", "pollutant_estimates", "processing_fps"
    ]
    for k in expected_json_keys:
        assert k in summary_data

def test_pipeline_video_processing(tmp_path):
    vid_p = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(vid_p), fourcc, 25.0, (640, 480))
    for i in range(10):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        cv2.rectangle(frame, (100 + i*5, 100), (300 + i*5, 250), (50, 50, 200), -1)
        writer.write(frame)
    writer.release()

    pipeline = EndToEndPipeline(conf_threshold=0.10)
    out_dir = tmp_path / "video_predictions"
    summary = pipeline.process_video(vid_p, output_dir=out_dir, save_video=True)

    assert (out_dir / "vehicles.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "tracked_video.mp4").exists()
    assert summary["total_unique_vehicles"] >= 0
