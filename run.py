"""
AI Vehicle Pollution Monitor - Main Unified Entrypoint CLI

Usage:
  python run.py --input data/raw/videos/sample_traffic.mp4
  python run.py --input data/raw/images/traffic_sample_1.jpg --output outputs/predictions
  python run.py --dashboard
  python run.py --test
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ensure UTF-8 output encoding for Windows CLI
os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline import EndToEndPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AI Vehicle Classification & Pollution Monitoring CLI")
    parser.add_argument("--input", type=str, help="Path to input image or video file")
    parser.add_argument("--output", type=str, default="outputs/predictions", help="Directory to save predictions (vehicles.csv, summary.json)")
    parser.add_argument("--confidence", type=float, default=0.40, help="YOLO detection confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'mps', 'auto')")
    parser.add_argument("--save-video", action="store_true", default=True, help="Save annotated video stream output")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--classifier-weights", type=str, default="models/classification/vehicle_classifier_best.pth", help="Classifier model path")
    parser.add_argument("--dashboard", "--app", dest="dashboard", action="store_true", help="Launch Streamlit web application dashboard")
    parser.add_argument("--prepare-data", action="store_true", help="Generate sample dataset and raw data files")
    parser.add_argument("--test", action="store_true", help="Run full system PyTest unit test suite")

    args = parser.parse_args()

    # 1. Launch Streamlit Dashboard
    if args.dashboard:
        logger.info("Launching Streamlit dashboard application...")
        cmd = [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"]
        subprocess.run(cmd)
        return

    # 2. Run Test Suite
    if args.test:
        logger.info("Running system PyTest suite...")
        cmd = [sys.executable, "-m", "pytest", "tests/"]
        subprocess.run(cmd)
        return

    # 3. Generate Sample Data
    if args.prepare_data:
        logger.info("Preparing sample data and dataset metadata...")
        from scripts.prepare_sample_data import main as prep_sample
        from scripts.prepare_datasets import main as prep_data
        prep_sample()
        prep_data()
        return

    # 4. Run End-to-End Pipeline on Input Image/Video
    if args.input:
        input_p = Path(args.input)
        if not input_p.exists():
            logger.error(f"Input file not found: {input_p}")
            sys.exit(1)

        pipeline = EndToEndPipeline(
            yolo_weights=args.weights,
            classifier_weights=args.classifier_weights,
            conf_threshold=args.confidence,
            device_pref=args.device
        )

        ext = input_p.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            logger.info(f"Processing image file: {input_p}")
            summary = pipeline.process_image(input_p, output_dir=args.output)
        elif ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
            logger.info(f"Processing video file: {input_p}")
            summary = pipeline.process_video(input_p, output_dir=args.output, save_video=args.save_video)
        else:
            logger.error(f"Unsupported file format '{ext}'. Choose an image or video file.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("END-TO-END PIPELINE PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Unique Vehicles   : {summary['total_unique_vehicles']}")
        print(f"Petrol Count            : {summary['petrol_count']}")
        print(f"Diesel Count            : {summary['diesel_count']}")
        print(f"EV Count                : {summary['ev_count']}")
        print(f"CNG/LPG Count           : {summary['cng_lpg_count']}")
        print(f"Hybrid Count            : {summary['hybrid_count']}")
        print(f"Unknown Count           : {summary['unknown_count']}")
        print(f"Pollution Index         : {summary['pollution_index']['vehicle_pollution_index']} / 100 ({summary['pollution_index']['category']})")
        print(f"Noise Index             : {summary['noise_index']['relative_noise_index']} / 100 ({summary['noise_index']['category']})")
        print(f"Processing Speed        : {summary['processing_fps']} FPS")
        print(f"Saved Vehicles CSV      : {Path(args.output) / 'vehicles.csv'}")
        print(f"Saved Summary JSON      : {Path(args.output) / 'summary.json'}")
        print("=" * 60 + "\n")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
