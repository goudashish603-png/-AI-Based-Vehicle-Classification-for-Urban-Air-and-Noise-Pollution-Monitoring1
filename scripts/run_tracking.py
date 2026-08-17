"""
Vehicle Tracking & Counting CLI Utility

Runs vehicle detection and multi-object tracking on prerecorded videos, 
tracks unique vehicle IDs, monitors virtual line crossings, and exports tracking_results.csv.
"""
import cv2
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detection.detector import VehicleDetector
from src.tracking.tracker import VehicleTracker
from src.tracking.line_counter import VirtualCountingLine
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Vehicle Tracking & Flow Counting CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file or stream")
    parser.add_argument("--output", type=str, default="outputs/predictions/tracked_video.mp4", help="Output annotated video path")
    parser.add_argument("--csv", type=str, default="outputs/predictions/tracking_results.csv", help="Output tracking_results.csv path")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.40, help="Detection confidence threshold")
    parser.add_argument("--line-y", type=int, default=360, help="Y-coordinate for horizontal virtual counting line")

    args = parser.parse_args()

    input_path = str(args.input)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detector = VehicleDetector(model_path=args.weights, conf_threshold=args.conf)
    counting_line = VirtualCountingLine(
        line_start=(0, args.line_y),
        line_end=(width, args.line_y),
        name="Traffic Line"
    )
    tracker = VehicleTracker(fps=fps, counting_line=counting_line)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

    frame_idx = 0
    logger.info(f"Starting vehicle tracking on {input_path} ({width}x{height} @ {fps} fps)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect & Track
        detections = detector.detect(frame)
        tracked_dets = tracker.update(detections, frame_idx=frame_idx)

        # Draw Overlay
        annotated = tracker.draw_tracking_overlay(frame, show_counting_line=True, show_trajectories=True)
        writer.write(annotated)

        frame_idx += 1

    cap.release()
    writer.release()

    # Export tracking_results.csv
    tracker.export_tracking_results(args.csv)

    density = tracker.get_active_density()
    print("\n" + "=" * 60)
    print("VEHICLE TRACKING & COUNTING SUMMARY REPORT")
    print("=" * 60)
    print(f"Processed Frames        : {frame_idx}")
    print(f"Total Unique Vehicles   : {density['total_unique_vehicles']}")
    print(f"Counting Line Crossings : {density['line_crossing_counts']['total']}")
    print(f"Annotated Video Output  : {out_p}")
    print(f"Tracking Results CSV    : {args.csv}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
