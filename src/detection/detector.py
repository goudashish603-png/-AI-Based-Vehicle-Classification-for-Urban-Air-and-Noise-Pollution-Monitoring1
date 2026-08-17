import sys
import types

# Patch missing _bz2 on Windows embedded Python environments
if '_bz2' not in sys.modules:
    try:
        import _bz2
    except ImportError:
        dummy_bz2 = types.ModuleType('_bz2')
        dummy_bz2.BZ2Compressor = object
        dummy_bz2.BZ2Decompressor = object
        sys.modules['_bz2'] = dummy_bz2

try:
    import cv2
except ImportError:
    cv2 = None

import time
import datetime
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

from src.detection.types import BoundingBox, DetectionResult
from src.detection.preprocessing import load_image
from src.detection.postprocessing import map_coco_class_name, export_detections_to_csv, export_detections_to_json
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DetectionBatchResult:
    """Dataclass holding detection output results for a video/image batch."""
    def __init__(self, detections: List[DetectionResult], frame_idx: int = 0, timestamp: str = "", inference_fps: float = 0.0):
        self.detections = detections
        self.frame_idx = frame_idx
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.inference_fps = inference_fps

    def to_dict_list(self) -> List[Dict[str, Any]]:
        records = []
        for det in self.detections:
            x1, y1, x2, y2 = det.bbox.to_int_tuple()
            records.append({
                "frame_idx": self.frame_idx,
                "timestamp": self.timestamp,
                "track_id": det.track_id if det.track_id is not None else -1,
                "class_name": det.class_name,
                "class_id": det.class_id,
                "confidence": round(det.confidence, 4),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "fuel_type": det.fuel_type or "Unknown"
            })
        return records


class VehicleDetector:
    """
    Production Vehicle Detection Engine wrapping YOLO (Ultralytics v8 / PyTorch Torchvision).
    Supports single image inference, full video stream inference, and webcam RTSP streams.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.sys_config = config or load_config()
        det_cfg = self.sys_config.get("detection", {})

        self.conf_threshold = conf_threshold if conf_threshold is not None else det_cfg.get("conf_threshold", 0.45)
        self.iou_threshold = iou_threshold if iou_threshold is not None else det_cfg.get("iou_threshold", 0.45)
        
        device_pref = self.sys_config.get("system", {}).get("device", "auto")
        self.device = get_device(device_pref)

        self.model_path = model_path or det_cfg.get("model_name", "yolov8n.pt")
        self.yolo_model = None
        self.torchvision_model = None

        self._init_model()

    def _init_model(self):
        """Initializes YOLO or FasterRCNN model."""
        # 1. Try Ultralytics YOLO
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.yolo_model = YOLO(self.model_path)
            logger.info("YOLO detector initialized successfully.")
            return
        except Exception as e:
            logger.warning(f"Ultralytics YOLO model failed to load ({e}). Using PyTorch Torchvision fallback...")

        # 2. PyTorch Torchvision FasterRCNN fallback
        try:
            import torchvision.models.detection as detection
            logger.info("Loading PyTorch Torchvision FasterRCNN model...")
            weights = detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.torchvision_model = detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.torchvision_model.to(self.device)
            self.torchvision_model.eval()
            logger.info("Torchvision FasterRCNN detector initialized successfully.")
        except Exception as e:
            logger.warning(f"Torchvision detection model failed ({e}). Operating with contour blob detector fallback.")

    def detect(self, frame: np.ndarray, conf_thresh: Optional[float] = None) -> List[DetectionResult]:
        """Runs vehicle detection on a single frame and returns list of DetectionResult objects."""
        batch_res = self.predict_frame(frame, conf_thresh=conf_thresh)
        return batch_res.detections

    def predict_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: Optional[str] = None,
        conf_thresh: Optional[float] = None
    ) -> DetectionBatchResult:
        """
        Runs object detection on a single BGR OpenCV image frame.
        
        Returns:
            DetectionBatchResult instance
        """
        if frame is None or frame.size == 0:
            return DetectionBatchResult([], frame_idx=frame_idx, timestamp=timestamp)

        conf = conf_thresh if conf_thresh is not None else self.conf_threshold
        ts = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        start_time = time.time()

        if self.yolo_model is not None:
            results = self._predict_yolo(frame, conf)
        elif self.torchvision_model is not None:
            results = self._predict_torchvision(frame, conf)
        else:
            results = self._predict_contour_fallback(frame, conf)

        fps = 1.0 / max(1e-5, time.time() - start_time)

        return DetectionBatchResult(results, frame_idx=frame_idx, timestamp=ts, inference_fps=fps)

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
        conf_thresh: Optional[float] = None
    ) -> Tuple[np.ndarray, DetectionBatchResult]:
        """
        Runs vehicle detection on an image file or numpy array.
        
        Returns:
            Tuple of (annotated_bgr_image, DetectionBatchResult)
        """
        frame = load_image(image_input)
        batch_res = self.predict_frame(frame, conf_thresh=conf_thresh)

        annotated = frame.copy()
        for det in batch_res.detections:
            x1, y1, x2, y2 = det.bbox.to_int_tuple()
            label = f"{det.class_name.upper()} {int(det.confidence*100)}%"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), annotated)
            logger.info(f"Saved annotated image to {out_p}")

        return annotated, batch_res

    def predict_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        conf_thresh: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[DetectionBatchResult]:
        """
        Runs vehicle detection on a video stream file or RTSP stream.
        
        Returns:
            List of DetectionBatchResult for all processed video frames.
        """
        v_path = str(video_path)
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {v_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        writer = None
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

        batch_results = []
        frame_idx = 0

        logger.info(f"Starting video inference on {v_path} ({width}x{height} @ {fps:.1f} fps)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames is not None and frame_idx >= max_frames):
                break

            batch_res = self.predict_frame(frame, frame_idx=frame_idx, conf_thresh=conf_thresh)
            batch_results.append(batch_res)

            if writer:
                annotated = frame.copy()
                for det in batch_res.detections:
                    x1, y1, x2, y2 = det.bbox.to_int_tuple()
                    label = f"{det.class_name.upper()} {int(det.confidence*100)}%"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, label, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Saved annotated video output to {output_path}")

        return batch_results

    def _predict_yolo(self, frame: np.ndarray, conf_thresh: float) -> List[DetectionResult]:
        results = []
        h, w = frame.shape[:2]
        # Target COCO vehicle class IDs: 1 (bicycle), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        vehicle_coco_classes = [1, 2, 3, 5, 7]
        try:
            preds = self.yolo_model(frame, imgsz=640, conf=conf_thresh, iou=self.iou_threshold, classes=vehicle_coco_classes, verbose=False)[0]
        except Exception:
            preds = self.yolo_model(frame, imgsz=640, conf=conf_thresh, iou=self.iou_threshold, verbose=False)[0]

        for box in preds.boxes:
            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())
            
            # Map COCO or custom YOLO class name
            raw_name = str(self.yolo_model.names.get(cls_id, "vehicle"))
            norm_class_name = map_coco_class_name(cls_id, raw_name)

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)

            bbox = BoundingBox(x1, y1, x2, y2)
            ix1, iy1, ix2, iy2 = bbox.to_int_tuple()
            crop = frame[iy1:iy2, ix1:ix2].copy() if (ix2 > ix1 and iy2 > iy1) else None

            results.append(DetectionResult(
                bbox=bbox,
                class_id=cls_id,
                class_name=norm_class_name,
                confidence=conf,
                vehicle_crop=crop
            ))
        return results

    def _predict_torchvision(self, frame: np.ndarray, conf_thresh: float) -> List[DetectionResult]:
        results = []
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.torchvision_model(tensor)[0]

        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score < conf_thresh:
                continue
            norm_class_name = map_coco_class_name(int(label))
            x1, y1, x2, y2 = map(float, box)
            bbox = BoundingBox(x1, y1, x2, y2)
            ix1, iy1, ix2, iy2 = bbox.to_int_tuple()
            crop = frame[iy1:iy2, ix1:ix2].copy() if (ix2 > ix1 and iy2 > iy1) else None

            results.append(DetectionResult(
                bbox=bbox,
                class_id=int(label),
                class_name=norm_class_name,
                confidence=float(score),
                vehicle_crop=crop
            ))
        return results

    def _predict_contour_fallback(self, frame: np.ndarray, conf_thresh: float) -> List[DetectionResult]:
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3500:
                x, y, bw, bh = cv2.boundingRect(cnt)
                bbox = BoundingBox(float(x), float(y), float(x + bw), float(y + bh))
                crop = frame[y:y+bh, x:x+bw].copy()
                aspect = bw / float(bh)
                norm_class = "bus" if aspect > 1.8 else ("car" if aspect > 1.2 else "truck")
                results.append(DetectionResult(
                    bbox=bbox,
                    class_id=2,
                    class_name=norm_class,
                    confidence=0.75,
                    vehicle_crop=crop
                ))
        return results
