import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.classification.dataset import VehicleMakeModelDataset, get_transforms
from src.classification.model import VehicleClassifierNet
from src.classification.train import ClassifierTrainer
from src.classification.evaluate import ClassifierEvaluator
from src.classification.inference import VehicleMakeModelClassifier
from src.classification.explainability import GradCAM

def test_dataset_loader(tmp_path):
    img_p = tmp_path / "dummy_car.jpg"
    img = Image.new("RGB", (100, 100), (200, 100, 50))
    img.save(img_p)

    df = pd.DataFrame([{
        "image_path": str(img_p),
        "manufacturer": "Tesla",
        "model": "Model 3",
        "label": "Tesla_Model 3"
    }])

    ds = VehicleMakeModelDataset(df, transform=get_transforms(is_train=False))
    assert len(ds) == 1
    tensor, cls_idx, label = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label == "Tesla_Model 3"

def test_classifier_network():
    net_resnet = VehicleClassifierNet(num_classes=5, backbone="resnet50", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    out_resnet = net_resnet(dummy_input)
    assert out_resnet.shape == (2, 5)

    net_eff = VehicleClassifierNet(num_classes=5, backbone="efficientnet_b0", pretrained=False)
    out_eff = net_eff(dummy_input)
    assert out_eff.shape == (2, 5)

def test_classifier_inference_api():
    classifier = VehicleMakeModelClassifier()
    dummy_crop = np.ones((100, 100, 3), dtype=np.uint8) * 180

    # predict single image
    mfr, model, conf, top_k = classifier.predict(dummy_crop, top_k=3)
    assert isinstance(mfr, str)
    assert isinstance(model, str)
    assert 0.0 <= conf <= 1.0
    assert len(top_k) == 3

    # predict_batch
    batch_res = classifier.predict_batch([dummy_crop, dummy_crop])
    assert len(batch_res) == 2

    # explain Grad-CAM
    orig, overlay = classifier.explain(dummy_crop, alpha=0.5)
    assert orig.shape == dummy_crop.shape
    assert overlay.shape == dummy_crop.shape

def test_classifier_evaluator(tmp_path):
    img_p = tmp_path / "car.jpg"
    Image.new("RGB", (100, 100)).save(img_p)
    df = pd.DataFrame([{"image_path": str(img_p), "label": "Tesla_Model 3"}])
    
    ds = VehicleMakeModelDataset(df, transform=get_transforms(is_train=False))
    loader = DataLoader(ds, batch_size=1)
    
    net = VehicleClassifierNet(num_classes=len(ds.class_to_idx), backbone="resnet50", pretrained=False)
    evaluator = ClassifierEvaluator(net, loader, ds.class_to_idx)
    metrics = evaluator.evaluate(output_dir=tmp_path / "reports")
    
    assert "top1_accuracy" in metrics
    assert "top5_accuracy" in metrics
    assert "precision_macro" in metrics
