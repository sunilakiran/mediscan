"""
preprocess.py
Handles all data loading, cleaning, and transformation
for chest X-ray images before training and inference.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split


# ── Constants ──────────────────────────────────────────
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet mean
STD  = [0.229, 0.224, 0.225]   # ImageNet std
CLASSES = ["NORMAL", "PNEUMONIA"]


# ── Transforms ─────────────────────────────────────────
def get_train_transforms():
    """
    Augmentation for training set.
    Random flips + rotation to prevent overfitting.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms():
    """
    No augmentation for validation and test sets.
    Only resize and normalize.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_inference_transforms():
    """
    Same as val — used at inference time in the API.
    """
    return get_val_transforms()


# ── Dataset Class ───────────────────────────────────────
class ChestXRayDataset(Dataset):
    """
    Custom Dataset for chest X-ray images.
    Expects folder structure:
        root/
          NORMAL/
          PNEUMONIA/
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []
        self.labels    = []

        for label_idx, class_name in enumerate(CLASSES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append(img_file)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label    = self.labels[idx]

        # Open and convert to RGB (some X-rays are grayscale)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ── DataLoaders ─────────────────────────────────────────
def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
):
    """
    Returns train, validation, and test DataLoaders.

    Note: The original val set has only 16 images — too
    small to be useful. We split the train set 80/20
    instead and use the original test set for final eval.
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    # Full train dataset with augmentation
    full_train = ChestXRayDataset(
        train_dir, transform=get_train_transforms()
    )

    # Split into train and validation
    total      = len(full_train)
    val_size   = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Override val transform — no augmentation
    val_dataset.dataset.transform = get_val_transforms()

    # Test dataset
    test_dataset = ChestXRayDataset(
        test_dir, transform=get_val_transforms()
    )

    # Class counts for imbalance logging
    normal_count    = full_train.labels.count(0)
    pneumonia_count = full_train.labels.count(1)
    print(f"[preprocess] NORMAL:    {normal_count}")
    print(f"[preprocess] PNEUMONIA: {pneumonia_count}")
    print(f"[preprocess] Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_class_weights(data_dir: str):
    """
    Compute class weights to handle imbalance.
    PNEUMONIA images are 3x more — we penalize
    majority class during training.
    """
    train_dir = os.path.join(data_dir, "train")
    dataset   = ChestXRayDataset(train_dir)

    normal_count    = dataset.labels.count(0)
    pneumonia_count = dataset.labels.count(1)
    total           = len(dataset)

    weight_normal    = total / (2 * normal_count)
    weight_pneumonia = total / (2 * pneumonia_count)

    weights = torch.tensor(
        [weight_normal, weight_pneumonia],
        dtype=torch.float,
    )
    print(f"[preprocess] Class weights → NORMAL: {weight_normal:.3f} | PNEUMONIA: {weight_pneumonia:.3f}")
    return weights


def preprocess_single_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a single PIL image for API inference.
    Returns a tensor of shape (1, 3, 224, 224).
    """
    transform = get_inference_transforms()
    tensor    = transform(image.convert("RGB"))
    return tensor.unsqueeze(0)  # Add batch dimension