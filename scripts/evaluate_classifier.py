"""
PyTorch Vehicle Make/Model Classifier Evaluation Script

Computes Top-1, Top-5 accuracy, macro/weighted precision, recall, F1 scores,
and exports confusion matrix plots and metrics JSON.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.classification.dataset import VehicleMakeModelDataset, get_transforms
from src.classification.model import VehicleClassifierNet
from src.classification.evaluate import ClassifierEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Vehicle Make/Model Classifier")
    parser.add_argument("--metadata", type=str, default="data/processed/unified_dataset_metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--weights", type=str, default="models/classification/vehicle_classifier_best.pth", help="Model weights path")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Model backbone name")
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Directory to save evaluation reports")

    args = parser.parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        logger.error(f"Metadata file {meta_path} not found. Run dataset pipeline first.")
        return

    df = pd.read_csv(meta_path)
    test_df = df[df.get("split", "train") == "test"]
    if test_df.empty:
        test_df = df

    test_dataset = VehicleMakeModelDataset(test_df, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = VehicleClassifierNet(num_classes=test_dataset.num_classes, backbone=args.backbone, pretrained=True)
    w_path = Path(args.weights)
    if w_path.exists():
        import torch
        state_dict = torch.load(w_path, map_location="cpu")
        model.load_state_dict(state_dict)

    evaluator = ClassifierEvaluator(
        model=model,
        test_loader=test_loader,
        class_to_idx=test_dataset.class_to_idx
    )

    metrics = evaluator.evaluate(output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("VEHICLE MAKE/MODEL CLASSIFIER EVALUATION REPORT")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:<25}: {v}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
