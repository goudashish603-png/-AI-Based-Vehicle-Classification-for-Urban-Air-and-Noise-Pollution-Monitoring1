"""
Run Vehicle Detection CLI Utility

Runs vehicle detection on images, videos, or live streams and exports telemetry logs.
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detection.detector import VehicleDetector
from src.detection.postprocessing import export_detections_to_csv, export_detections_to_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection CLI Engine")
    parser.add_argument("--input", type=str, required=True, help="Path to input image/video or stream index (0)")
    parser.add_argument("--output", type=str, default=None, help="Path to output annotated image/video file")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights file path or model name")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold (0.0 to 1.0)")
    parser.add_argument("--csv", type=str, default=None, help="Path to export detections CSV")
    parser.add_argument("--json", type=str, default=None, help="Path to export detections JSON")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process for video")

    args = parser.parse_args()

    input_path = Path(args.input) if not args.input.isdigit() else args.input
    detector = VehicleDetector(
        model_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    all_records = []

    if isinstance(input_path, Path) and input_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        logger.info(f"Processing image: {input_path}")
        annotated_img, batch_res = detector.predict_image(input_path, save_path=args.output, conf_thresh=args.conf)
        all_records.extend(batch_res.to_dict_list())
        print(f"Detected {len(batch_res.detections)} vehicles in image.")

    else:
        # Process Video or Camera Stream
        logger.info(f"Processing video/stream: {input_path}")
        batch_results = detector.predict_video(
            video_path=str(input_path),
            output_path=args.output,
            conf_thresh=args.conf,
            max_frames=args.max_frames
        )
        for b_res in batch_results:
            all_records.extend(b_res.to_dict_list())
        print(f"Processed {len(batch_results)} video frames. Total detections: {len(all_records)}")

    # Export telemetry logs
    if args.csv:
        export_detections_to_csv(all_records, args.csv)
    if args.json:
        export_detections_to_json(all_records, args.json)

if __name__ == "__main__":
    main()
