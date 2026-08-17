import os
try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False

from src.utils.config import load_model_config, load_config
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

FUEL_CLASSES = [
    "Petrol",
    "Diesel",
    "Electric Vehicle (EV)",
    "CNG/LPG",
    "Hybrid",
    "Unknown"
]

class FineGrainedVehicleNet(nn.Module):
    """
    Fine-grained PyTorch classification network based on ResNet50 / EfficientNet backbone.
    """
    def __init__(self, num_classes: int = 6, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            # Custom fine-grained classifier head
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class VehicleClassifier:
    """
    Inference and prediction engine for fine-grained vehicle & fuel-type classification.
    """
    def __init__(self, weights_path: Optional[Path] = None, config: Optional[Dict] = None):
        self.sys_config = load_config()
        self.model_config = load_model_config()
        
        self.device = get_device(self.sys_config.get("system", {}).get("device", "auto"))
        self.fuel_classes = self.model_config.get("model", {}).get("classes", {
            0: "Petrol", 1: "Diesel", 2: "Electric Vehicle (EV)",
            3: "CNG/LPG", 4: "Hybrid", 5: "Unknown"
        })
        # Format map keys to int if needed
        self.class_map = {int(k): v for k, v in self.fuel_classes.items()}
        
        backbone_name = self.model_config.get("model", {}).get("architecture", "resnet50")
        num_classes = len(self.class_map)
        
        self.model = FineGrainedVehicleNet(num_classes=num_classes, backbone=backbone_name, pretrained=True)
        
        # Load weights if custom trained file exists
        ckpt_path = weights_path or (Path(__file__).resolve().parent.parent.parent / "models" / "classification" / "vehicle_classifier.pth")
        if ckpt_path.exists():
            logger.info(f"Loading classifier weights from {ckpt_path}")
            try:
                state_dict = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                logger.warning(f"Could not load custom weights ({e}). Operating with initialized baseline.")
        else:
            logger.info("Using baseline fine-grained classifier network.")

        self.model.to(self.device)
        self.model.eval()

        # Image Preprocessing Pipeline
        img_size = self.model_config.get("data", {}).get("input_size", [224, 224])
        self.transform = transforms.Compose([
            transforms.Resize(tuple(img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.model_config.get("data", {}).get("mean", [0.485, 0.456, 0.406]),
                std=self.model_config.get("data", {}).get("std", [0.229, 0.224, 0.225])
            )
        ])

    def predict_crop(self, vehicle_crop: np.ndarray, vehicle_class_hint: str = "car") -> Tuple[str, float, Dict[str, float]]:
        """
        Classifies cropped vehicle image into fuel category.
        
        Returns:
            Tuple of (predicted_fuel_type, confidence_score, probabilities_dict)
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "Unknown", 0.0, {cls: 0.0 for cls in FUEL_CLASSES}

        # Convert OpenCV BGR image to PIL RGB
        try:
            rgb_img = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            pred_fuel = self.class_map.get(top_idx, "Unknown")

            # Map probabilities dict
            prob_dict = {self.class_map.get(i, f"Class_{i}"): float(probs[i]) for i in range(len(probs))}

            return pred_fuel, top_conf, prob_dict
        except Exception as e:
            logger.error(f"Error during classifier inference: {e}")
            return "Unknown", 0.0, {cls: 0.0 for cls in FUEL_CLASSES}
