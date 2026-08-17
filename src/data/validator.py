import os
import hashlib
import cv2
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

class DatasetValidator:
    """
    Validation engine for dataset integrity:
    - Corrupt image detection
    - MD5 hash duplicate detection
    - Bounding box annotation sanity checks
    - Train/Test split leakage verification
    """
    
    @staticmethod
    def is_valid_image(image_path: Path) -> bool:
        """Verifies whether an image can be opened and decoded without corruption."""
        if not image_path.exists() or image_path.stat().st_size == 0:
            return False
        try:
            with Image.open(image_path) as img:
                img.verify()  # PIL verify
            # Secondary OpenCV check to ensure valid pixel array decode
            cv_img = cv2.imread(str(image_path))
            return cv_img is not None and cv_img.size > 0
        except Exception:
            return False

    @staticmethod
    def compute_image_hash(image_path: Path) -> str:
        """Computes MD5 hash of an image file for duplicate detection."""
        hasher = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def find_duplicate_images(image_paths: List[Path]) -> Dict[str, List[Path]]:
        """
        Scans a list of image paths and groups identical duplicates by MD5 hash.
        
        Returns:
            Dict mapping MD5 hash -> list of duplicate image paths
        """
        hash_map: Dict[str, List[Path]] = {}
        for p in image_paths:
            if DatasetValidator.is_valid_image(p):
                h = DatasetValidator.compute_image_hash(p)
                hash_map.setdefault(h, []).append(p)

        # Filter only hashes with duplicates (> 1 file)
        duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
        return duplicates

    @staticmethod
    def check_split_leakage(train_paths: List[Path], test_paths: List[Path]) -> Tuple[bool, List[str]]:
        """
        Checks for data leakage between train and test splits based on image MD5 hashes.
        
        Returns:
            Tuple of (has_leakage, list_of_leaked_hashes)
        """
        train_hashes = {DatasetValidator.compute_image_hash(p): p for p in train_paths if DatasetValidator.is_valid_image(p)}
        test_hashes = {DatasetValidator.compute_image_hash(p): p for p in test_paths if DatasetValidator.is_valid_image(p)}

        leaked_hashes = list(set(train_hashes.keys()).intersection(set(test_hashes.keys())))
        has_leakage = len(leaked_hashes) > 0
        
        if has_leakage:
            logger.warning(f"Detected DATA LEAKAGE: {len(leaked_hashes)} images exist in both train and test splits!")
            
        return has_leakage, leaked_hashes

    @staticmethod
    def validate_yolo_annotation(
        txt_path: Path,
        img_width: int,
        img_height: int,
        num_classes: int = 80
    ) -> Tuple[bool, List[str]]:
        """
        Validates YOLO format annotation file (`class_id x_center y_center width height`).
        """
        if not txt_path.exists():
            return False, ["Annotation file missing"]

        errors = []
        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Line {i+1}: expected 5 elements, got {len(parts)}")
                    continue

                cls_id_str, xc_str, yc_str, w_str, h_str = parts
                cls_id = int(cls_id_str)
                xc, yc, w, h = float(xc_str), float(yc_str), float(w_str), float(h_str)

                if cls_id < 0 or cls_id >= num_classes:
                    errors.append(f"Line {i+1}: class_id {cls_id} out of range [0, {num_classes-1}]")

                for val_name, val in [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]:
                    if val < 0.0 or val > 1.0:
                        errors.append(f"Line {i+1}: normalized coordinate {val_name}={val} outside [0.0, 1.0]")

            return len(errors) == 0, errors
        except Exception as e:
            return False, [f"Unparseable file ({e})"]
