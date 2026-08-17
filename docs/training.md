# Model Training & Evaluation Guide

## 1. Vehicle Detector Training (`scripts/train_detector.py`)

Train custom YOLO object detector on fine-grained traffic datasets:

```bash
python scripts/train_detector.py --data data/processed/unified_dataset_metadata.csv --epochs 50 --imgsz 640 --batch 16
```

## 2. Vehicle Make/Model Classifier Training (`scripts/train_classifier.py`)

Train PyTorch ResNet50 transfer learning classifier:

```bash
python scripts/train_classifier.py --data data/processed/unified_dataset_metadata.csv --backbone resnet50 --epochs 25 --batch 32 --lr 0.001
```

## 3. Classifier Evaluation (`scripts/evaluate_classifier.py`)

Compute Top-1, Top-5 accuracy, precision, recall, F1-score, and confusion matrix:

```bash
python scripts/evaluate_classifier.py --weights models/classification/vehicle_classifier_best.pth --data data/processed/unified_dataset_metadata.csv
```
