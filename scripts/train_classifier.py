"""
PyTorch Vehicle Make/Model Classifier Training Script

Fine-tunes ResNet50 / EfficientNet-B0 transfer learning backbone on vehicle crop metadata.
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
from src.classification.train import ClassifierTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Vehicle Make/Model Classifier")
    parser.add_argument("--metadata", type=str, default="data/processed/unified_dataset_metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"], help="CNN backbone")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--imgsz", type=int, default=224, help="Target image size")
    parser.add_argument("--checkpoint-dir", type=str, default="models/classification", help="Directory to save model checkpoints")

    args = parser.parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        logger.warning(f"Metadata file {meta_path} not found. Running dataset pipeline first...")
        from src.data.pipeline import DatasetPipeline
        pipeline = DatasetPipeline()
        df = pipeline.run_pipeline()
    else:
        df = pd.read_csv(meta_path)

    if df.empty:
        logger.error("Dataset DataFrame is empty. Cannot train classifier.")
        print("\nTo train on Stanford Cars or CompCars, place dataset files in data/raw/ and run:")
        print("python scripts/prepare_datasets.py")
        print("python scripts/train_classifier.py --epochs 15\n")
        return

    # Train / Val split
    train_df = df[df.get("split", "train") == "train"]
    val_df = df[df.get("split", "train") == "test"]
    if val_df.empty:
        val_df = train_df.sample(frac=0.2, random_state=42)

    transforms_train = get_transforms(input_size=(args.imgsz, args.imgsz), is_train=True)
    transforms_val = get_transforms(input_size=(args.imgsz, args.imgsz), is_train=False)

    train_dataset = VehicleMakeModelDataset(train_df, transform=transforms_train)
    val_dataset = VehicleMakeModelDataset(val_df, transform=transforms_val, class_to_idx=train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples over {train_dataset.num_classes} vehicle make/model classes.")

    model = VehicleClassifierNet(num_classes=train_dataset.num_classes, backbone=args.backbone, pretrained=True)
    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )

    history = trainer.fit(epochs=args.epochs)
    print("\nClassifier training completed successfully!")
    print(f"Model saved to: {Path(args.checkpoint_dir) / 'vehicle_classifier_best.pth'}\n")

if __name__ == "__main__":
    main()
