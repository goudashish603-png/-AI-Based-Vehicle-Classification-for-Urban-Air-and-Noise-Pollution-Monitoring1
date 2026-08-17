import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from src.classification.model import VehicleClassifierNet
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EarlyStopping:
    """Early stopping handler to terminate training when validation loss stops improving."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ClassifierTrainer:
    """
    Production Training Pipeline for Fine-Grained Vehicle Classifier.
    """
    def __init__(
        self,
        model: VehicleClassifierNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0001,
        patience: int = 5,
        checkpoint_dir: Union[str, Path] = "models/classification",
        device_pref: str = "auto"
    ):
        self.device = get_device(device_pref)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.early_stopping = EarlyStopping(patience=patience)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    def fit(self, epochs: int = 10) -> Dict[str, List[float]]:
        logger.info(f"Starting model training for {epochs} epochs on device: {self.device}")
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            dt = time.time() - t0

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{dt:.1f}s] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = self.checkpoint_dir / "vehicle_classifier_best.pth"
                torch.save(self.model.state_dict(), best_path)
                logger.info(f"Saved new best model checkpoint to {best_path}")

            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Export training history JSON
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
