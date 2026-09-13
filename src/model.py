"""
Model training: Fine-tune ViT-B/16 on iNaturalist 2021.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score


class ImageClassifier(nn.Module):
    """ViT-B/16 classifier for iNaturalist."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # Load ViT-B/16 from timm
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        """Extract features before classification head."""
        # Access the backbone's forward pass up to classification
        return self.backbone.forward_head(self.backbone.forward_features(x))


class Trainer:
    """Training loop for image classification."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="VALIDATION")
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        accuracy = 100. * correct / total
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        save_dir: str = "models",
        patience: int = 5,
    ):
        """Full training loop with early stopping."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Learning rate scheduler
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        patience_counter = 0
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_acc'].append(train_metrics['accuracy'])

            # Validate
            val_metrics = self.validate(val_loader)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_acc'].append(val_metrics['accuracy'])

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                model_path = Path(save_dir) / "best_model.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"  ✓ Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            scheduler.step()

        return training_history

    def evaluate_head_tail(self, head_loader, tail_loader):
        """Evaluate accuracy on head and tail classes separately."""
        self.model.eval()

        print("\n" + "="*60)
        print("HEAD vs TAIL CLASS EVALUATION")
        print("="*60)

        # Head classes
        head_correct = 0
        head_total = 0
        with torch.no_grad():
            for images, labels, _ in tqdm(head_loader, desc="HEAD"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                head_correct += predicted.eq(labels).sum().item()
                head_total += labels.size(0)

        head_acc = 100. * head_correct / head_total

        # Tail classes
        tail_correct = 0
        tail_total = 0
        with torch.no_grad():
            for images, labels, _ in tqdm(tail_loader, desc="TAIL"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                tail_correct += predicted.eq(labels).sum().item()
                tail_total += labels.size(0)

        tail_acc = 100. * tail_correct / tail_total

        print(f"\nHead Class Accuracy: {head_acc:.2f}% ({head_correct}/{head_total})")
        print(f"Tail Class Accuracy: {tail_acc:.2f}% ({tail_correct}/{tail_total})")
        print(f"Accuracy Gap: {head_acc - tail_acc:.2f}%")
        print("="*60 + "\n")

        return {
            'head_accuracy': head_acc,
            'tail_accuracy': tail_acc,
            'head_correct': head_correct,
            'head_total': head_total,
            'tail_correct': tail_correct,
            'tail_total': tail_total
        }

    def predict_with_confidence(self, val_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions with confidence scores (before calibration).

        Returns:
            (predictions, confidences, labels)
        """
        self.model.eval()
        all_preds = []
        all_confs = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Getting predictions"):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = probs.max(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_confs.extend(confs.cpu().numpy())
                all_labels.extend(labels.numpy())

        return (
            np.array(all_preds),
            np.array(all_confs),
            np.array(all_labels)
        )
