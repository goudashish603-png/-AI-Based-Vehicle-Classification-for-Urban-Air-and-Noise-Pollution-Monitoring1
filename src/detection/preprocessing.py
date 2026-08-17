import cv2
import numpy as np
from typing import Tuple, Union
from pathlib import Path
from PIL import Image

def load_image(input_source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Loads an image from file path, PIL Image, or returns numpy array BGR frame.
    """
    if isinstance(input_source, (str, Path)):
        path_str = str(input_source)
        if not Path(path_str).exists():
            raise FileNotFoundError(f"Input image path does not exist: {path_str}")
        img = cv2.imread(path_str)
        if img is None:
            raise ValueError(f"Failed to decode image from path: {path_str}")
        return img
    elif isinstance(input_source, Image.Image):
        # Convert PIL to BGR OpenCV numpy array
        rgb = np.array(input_source)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(input_source, np.ndarray):
        return input_source
    else:
        raise TypeError(f"Unsupported image input type: {type(input_source)}")

def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resizes image with aspect ratio padding (letterboxing).
    
    Returns:
        Tuple of (padded_image, scale_factor, (pad_w, pad_h))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / float(w), target_h / float(h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    padded = cv2.copyMakeBorder(
        resized,
        top=pad_h,
        bottom=target_h - new_h - pad_h,
        left=pad_w,
        right=target_w - new_w - pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=fill_color
    )

    return padded, scale, (pad_w, pad_h)
