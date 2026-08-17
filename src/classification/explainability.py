import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Tuple, Optional
from torchvision import transforms

from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) engine for PyTorch CNNs.
    Explains model decisions by generating visual heatmaps on input vehicle crops.
    """
    def __init__(self, model: nn.Module, target_layer_name: Optional[str] = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Determine target layer (default: last conv layer of backbone)
        if target_layer_name is None:
            if hasattr(model, "backbone"):
                if hasattr(model.backbone, "layer4"):  # ResNet50
                    self.target_layer = model.backbone.layer4[-1]
                elif hasattr(model.backbone, "features"): # EfficientNet
                    self.target_layer = model.backbone.features[-1]
                else:
                    self.target_layer = list(model.backbone.children())[-2]
            else:
                self.target_layer = list(model.children())[-2]
        else:
            self.target_layer = dict(model.named_modules())[target_layer_name]

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generates a Grad-CAM heatmap array normalized [0, 1].
        
        Args:
            input_tensor: PyTorch tensor of shape (1, 3, H, W)
            target_class_idx: Optional index of class to generate CAM for
            
        Returns:
            Tuple of (cam_heatmap, target_class_idx)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class_idx]
        score.backward()

        gradients = self.gradients[0].cpu().data.numpy()  # (C, H, W)
        activations = self.activations[0].cpu().data.numpy()  # (C, H, W)

        # Global average pooling over gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted combination of forward activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU on CAM to isolate positive influential features
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class_idx

    def overlay_heatmap(
        self,
        vehicle_crop_bgr: np.ndarray,
        cam_heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlays the CAM heatmap onto the original OpenCV BGR vehicle crop.
        """
        if vehicle_crop_bgr is None or vehicle_crop_bgr.size == 0:
            return vehicle_crop_bgr

        h, w = vehicle_crop_bgr.shape[:2]
        resized_cam = cv2.resize(cam_heatmap, (w, h))

        # Convert to 8-bit color map (JET)
        heatmap_uint8 = np.uint8(255 * resized_cam)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Superimpose
        overlay = cv2.addWeighted(vehicle_crop_bgr, 1.0 - alpha, color_heatmap, alpha, 0)
        return overlay
