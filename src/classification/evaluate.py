import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from src.classification.model import VehicleClassifierNet
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ClassifierEvaluator:
    """
    Evaluation Engine for Fine-Grained Vehicle Classifier.
    Computes Top-1, Top-5 Accuracy, Precision, Recall, F1, Confusion Matrix.
    """
    def __init__(
        self,
        model: VehicleClassifierNet,
        test_loader: DataLoader,
        class_to_idx: Dict[str, int],
        device_pref: str = "auto"
    ):
        self.device = get_device(device_pref)
        self.model = model.to(self.device)
        self.model.eval()
        self.test_loader = test_loader
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes = len(class_to_idx)

    def evaluate(self, output_dir: Union[str, Path] = "outputs/reports") -> Dict[str, Any]:
        """
        Runs evaluation over test loader and returns evaluation metrics dictionary.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                preds = torch.argmax(outputs, dim=1)

                all_targets.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        total_samples = len(all_targets)
        if total_samples == 0:
            logger.warning("Test loader was empty.")
            return {}

        # Top-1 Accuracy
        top1_correct = (all_preds == all_targets).sum()
        top1_acc = float(top1_correct / total_samples)

        # Top-5 Accuracy
        top5_correct = 0
        for i in range(total_samples):
            top_k_indices = np.argsort(all_probs[i])[-min(5, self.num_classes):]
            if all_targets[i] in top_k_indices:
                top5_correct += 1
        top5_acc = float(top5_correct / total_samples)

        # Precision, Recall, F1
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

        # Classification Report
        target_names = [self.idx_to_class.get(i, f"Class_{i}") for i in range(self.num_classes)]
        clf_report_str = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(self.num_classes)))

        metrics = {
            "total_samples": total_samples,
            "top1_accuracy": round(top1_acc, 4),
            "top5_accuracy": round(top5_acc, 4),
            "precision_macro": round(float(p_macro), 4),
            "recall_macro": round(float(r_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_weighted": round(float(p_weighted), 4),
            "recall_weighted": round(float(r_weighted), 4),
            "f1_weighted": round(float(f1_weighted), 4)
        }

        # Save Metrics JSON
        metrics_json_path = out_dir / "classifier_evaluation_metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save Classification Report text
        report_txt_path = out_dir / "classifier_classification_report.txt"
        with open(report_txt_path, "w") as f:
            f.write(clf_report_str)

        # Plot & Save Confusion Matrix
        self._plot_confusion_matrix(cm, target_names, out_dir / "classifier_confusion_matrix.png")

        logger.info(f"Evaluation complete! Top-1 Acc: {top1_acc*100:.2f}%, Top-5 Acc: {top5_acc*100:.2f}%")
        return metrics

    def _plot_confusion_matrix(self, cm: np.ndarray, target_names: List[str], save_path: Path):
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Classifier Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
