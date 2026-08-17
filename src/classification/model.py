import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

class VehicleClassifierNet(nn.Module):
    """
    PyTorch Transfer Learning Architecture for Fine-Grained Vehicle Make/Model Classification.
    Supports ResNet50 and EfficientNet-B0 backbones.
    """
    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = num_classes

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            # Replace final classification head
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate / 2.0),
                nn.Linear(512, num_classes)
            )
        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate / 2.0),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported architecture '{backbone}'. Choose 'resnet50' or 'efficientnet_b0'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Returns the target convolutional layer for Grad-CAM explainability."""
        if self.backbone_name == "resnet50":
            return self.backbone.layer4[-1]
        elif self.backbone_name == "efficientnet_b0":
            return self.backbone.features[-1]
        else:
            return list(self.backbone.children())[-2]
