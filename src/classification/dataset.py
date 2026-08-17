import os
import sys
import types

# Patch missing _bz2 on Windows embedded Python distributions
if '_bz2' not in sys.modules:
    try:
        import _bz2
    except ImportError:
        dummy_bz2 = types.ModuleType('_bz2')
        dummy_bz2.BZ2Compressor = object
        dummy_bz2.BZ2Decompressor = object
        sys.modules['_bz2'] = dummy_bz2

import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Any

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    transforms = None
    Dataset = object
    DataLoader = object

from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_transforms(
    input_size: Tuple[int, int] = (224, 224),
    is_train: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Any:
    """
    Returns image transformation pipeline for PyTorch training and inference.
    """
    if not TORCH_AVAILABLE or transforms is None:
        return None

    if is_train:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class VehicleMakeModelDataset(Dataset):
    """
    PyTorch Dataset for Fine-Grained Vehicle Make & Model Classification.
    Ingests metadata DataFrame or folder path.
    """
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[Any] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.df = metadata_df.copy().reset_index(drop=True)
        self.transform = transform or get_transforms(is_train=False)

        # Build make/model label: e.g. "BMW_3 Series"
        if "label" not in self.df.columns:
            self.df["label"] = self.df["manufacturer"].astype(str) + "_" + self.df["model"].astype(str)

        # Map class names to integer indices
        if class_to_idx is None:
            unique_classes = sorted(self.df["label"].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        label_str = str(row["label"])

        class_idx = self.class_to_idx.get(label_str, 0)

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback black image if corrupt
            pil_img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.ToTensor()(pil_img)

        return tensor_img, class_idx, label_str
