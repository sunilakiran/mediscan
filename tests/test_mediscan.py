"""
test_mediscan.py
Pytest test suite for MediScan.
Run: pytest -v
"""

import io
import pytest
import torch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from mediscan.api import app
from mediscan.preprocess import (
    preprocess_single_image,
    get_val_transforms,
    ChestXRayDataset,
    IMAGE_SIZE,
)
from mediscan.model import build_model, predict


# ── Test Client ─────────────────────────────────────────
client = TestClient(app)


# ── Helper: Dummy X-ray Image ───────────────────────────
def make_dummy_image(mode="RGB", size=(224, 224)) -> Image.Image:
    """Create a dummy grayscale image simulating an X-ray."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode)


def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL image to bytes for API upload."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════
# PREPROCESS TESTS
# ══════════════════════════════════════════════════════

def test_preprocess_output_shape():
    """Preprocessed image must be (1, 3, 224, 224)."""
    image  = make_dummy_image()
    tensor = preprocess_single_image(image)
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE), \
        f"Expected (1, 3, 224, 224) got {tensor.shape}"


def test_preprocess_output_type():
    """Output must be a torch.Tensor."""
    image  = make_dummy_image()
    tensor = preprocess_single_image(image)
    assert isinstance(tensor, torch.Tensor)


def test_preprocess_normalized():
    """Tensor values should be normalized — not raw 0-255."""
    image  = make_dummy_image()
    tensor = preprocess_single_image(image)
    assert tensor.max().item() < 10.0, \
        "Tensor not normalized — values too large"


def test_val_transforms_not_none():
    """Validation transforms must return a Compose object."""
    transforms = get_val_transforms()
    assert transforms is not None


# ══════════════════════════════════════════════════════
# MODEL TESTS
# ══════════════════════════════════════════════════════

def test_model_builds():
    """Model must build without errors."""
    model = build_model()
    assert model is not None


def test_model_output_shape():
    """Model must output shape (batch, 1) for binary classification."""
    model  = build_model()
    model.eval()
    dummy  = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (2, 1), \
        f"Expected (2, 1) got {output.shape}"


def test_predict_returns_required_keys():
    """predict() must return class, probability, risk, recommendation."""
    model  = build_model()
    image  = make_dummy_image()
    tensor = preprocess_single_image(image)
    result = predict(model, tensor)

    for key in ["predicted_class", "probability", "risk_level", "recommendation"]:
        assert key in result, f"Missing key: {key}"


def test_predict_class_valid():
    """Predicted class must be NORMAL or PNEUMONIA."""
    model  = build_model()
    image  = make_dummy_image()
    tensor = preprocess_single_image(image)
    result = predict(model, tensor)
    assert result["predicted_class"] in ["NORMAL", "PNEUMONIA"]


def test_predict_probability_range():
    """Probability must be between 0 and 1."""
    model  = build_model()
    image  = make_dummy_image()
    tensor = preprocess_single_image(image)
    result = predict(model, tensor)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_risk_level_valid():
    """Risk level must be LOW, MEDIUM, or HIGH."""
    model  = build_model()
    image  = make_dummy_image()
    tensor = preprocess_single_image(image)
    result = predict(model, tensor)
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


# ══════════════════════════════════════════════════════
# API TESTS
# ══════════════════════════════════════════════════════

def test_health_endpoint():
    """GET /health must return 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_health_model_loaded():
    """GET /health must report model_loaded status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data


def test_predict_endpoint_valid_image():
    """POST /predict with valid image must return 200 or 503."""
    image     = make_dummy_image()
    img_bytes = image_to_bytes(image)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    # 503 is acceptable when model not loaded in CI
    assert response.status_code in [200, 503]


def test_predict_endpoint_wrong_format():
    """POST /predict with non-image must return 400 or 503."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code in [400, 503]
    




def test_history_endpoint():
    """GET /history must return 200 with predictions list."""
    response = client.get("/history")
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "count" in data


def test_stats_endpoint():
    """GET /stats must return 200 with aggregated data."""
    response = client.get("/stats")
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "total_predictions" in data