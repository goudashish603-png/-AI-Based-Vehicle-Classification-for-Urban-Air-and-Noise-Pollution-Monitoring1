"""
YOLO Vehicle Detector Training & Fine-Tuning Script

Provides standard Ultralytics YOLO training pipeline for custom vehicle detection datasets.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_yolo(
    data_yaml: str,
    weights: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    project: str = "models/detection"
):
    """
    Fine-tunes YOLO model on custom dataset.yaml configuration.
    """
    try:
        from ultralytics import YOLO
        logger.info(f"Initializing YOLO model from base weights: {weights}")
        model = YOLO(weights)

        logger.info(f"Starting training on {data_yaml} for {epochs} epochs...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device if device != "auto" else 0,
            project=project,
            name="custom_vehicle_yolo",
            exist_ok=True
        )

        logger.info(f"Training completed successfully! Model saved to {project}/custom_vehicle_yolo/weights/best.pt")
        return results
    except Exception as e:
        logger.error(f"Error during detector training: {e}")
        print("\nNote: Ultralytics package is required for full YOLO training.")
        print("Install via: pip install ultralytics\n")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train Custom YOLO Vehicle Detector")
    parser.add_argument("--data", type=str, default="data/processed/ua_detrac/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Base weights path or model name")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Target image size")
    parser.add_argument("--device", type=str, default="auto", help="Device (0 for GPU, cpu for CPU)")

    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device
    )

if __name__ == "__main__":
    main()
