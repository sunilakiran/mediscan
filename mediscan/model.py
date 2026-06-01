"""
model.py
ResNet-50 fine-tuned for binary chest X-ray classification.
Exposes train(), predict(), evaluate() functions.
"""

import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from pathlib import Path


# ── Constants ───────────────────────────────────────────
MODEL_PATH = "mediscan_model.pt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES    = ["NORMAL", "PNEUMONIA"]


# ── Model Architecture ──────────────────────────────────
def build_model() -> nn.Module:
    """
    Load pretrained ResNet-50 and replace
    the final layer for binary classification.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer — only this will be trained initially
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 1),  # Binary output
    )

    return model.to(DEVICE)


# ── Train ───────────────────────────────────────────────
def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-3,
    experiment_name: str = "MediScan",
):
    """
    Train ResNet-50 with class-weighted loss.
    Logs all metrics to MLflow.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        model     = build_model()
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=class_weights[1].to(DEVICE)
        )
        optimizer = torch.optim.AdamW(
            model.fc.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        # Log hyperparameters
        mlflow.log_params({
            "epochs":        epochs,
            "lr":            lr,
            "batch_size":    train_loader.batch_size,
            "device":        str(DEVICE),
            "architecture":  "ResNet-50",
            "imbalance":     "class-weighted BCE loss",
        })

        best_val_auc = 0.0

        for epoch in range(epochs):
            # ── Training Phase ──
            model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            scheduler.step()

            # ── Validation Phase ──
            val_metrics = evaluate(model, val_loader)
            avg_loss    = train_loss / len(train_loader)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {avg_loss:.4f} | "
                f"Val AUC: {val_metrics['auroc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss":  avg_loss,
                "val_auroc":   val_metrics["auroc"],
                "val_f1":      val_metrics["f1"],
                "val_recall":  val_metrics["recall"],
                "val_precision": val_metrics["precision"],
            }, step=epoch)

            # Save best model
            if val_metrics["auroc"] > best_val_auc:
                best_val_auc = val_metrics["auroc"]
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  ✅ Best model saved — AUC: {best_val_auc:.4f}")

        # Log final model to MLflow
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(MODEL_PATH)
        print(f"\n[model] Training complete — Best AUC: {best_val_auc:.4f}")

    return model


# ── Evaluate ────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model on a DataLoader.
    Returns precision, recall, F1, AUROC for both classes.
    """
    model.eval()
    all_labels  = []
    all_probs   = []
    all_preds   = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs   = torch.sigmoid(outputs).squeeze(1).cpu()
            preds   = (probs >= threshold).long()

            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        output_dict=True,
    )

    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "auroc":     roc_auc_score(all_labels, all_probs),
        "report":    report,
    }

    return metrics


# ── Predict ─────────────────────────────────────────────
def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Predict on a single image tensor.
    Returns class, probability, and risk level.
    """
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        output       = model(image_tensor)
        prob         = torch.sigmoid(output).item()
        pred_class   = 1 if prob >= threshold else 0

    risk_level = (
        "HIGH"   if prob >= 0.75 else
        "MEDIUM" if prob >= 0.50 else
        "LOW"
    )

    recommendation = (
        "Immediate radiologist review recommended — high probability of pneumonia."
        if risk_level == "HIGH" else
        "Further evaluation advised — moderate pneumonia indicators detected."
        if risk_level == "MEDIUM" else
        "No significant pneumonia indicators detected. Routine follow-up advised."
    )

    return {
        "predicted_class": CLASSES[pred_class],
        "probability":     round(prob, 4),
        "risk_level":      risk_level,
        "recommendation":  recommendation,
    }


# ── Load Model ──────────────────────────────────────────
def load_model(model_path: str = MODEL_PATH) -> nn.Module:
    """
    Load saved model weights from disk.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    model = build_model()
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.eval()
    print(f"[model] Loaded from {model_path}")
    return model