"""
api.py
FastAPI application for MediScan.
Endpoints:
    GET  /health
    POST /predict
    GET  /history
    GET  /stats
"""

import io
import base64
import hashlib
from datetime import datetime, timezone

import torch
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

from mediscan.preprocess import preprocess_single_image
from mediscan.model import load_model, predict, DEVICE

load_dotenv()

# ── App Setup ───────────────────────────────────────────
app = FastAPI(
    title="MediScan API",
    description="AI-powered chest X-ray pneumonia detection",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MongoDB Setup with In-Memory Fallback ───────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = "mediscan"

in_memory_predictions = []
mongo_available = False

try:
    # PyMongo is lazy, so we run a ping command to check if the connection is active
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
    client.admin.command('ping')
    db = client[DB_NAME]
    predictions_col = db["predictions"]
    training_runs_col = db["training_runs"]
    mongo_available = True
    print("[api] MongoDB connected ✅")
except Exception as e:
    predictions_col = None
    training_runs_col = None
    print(f"[api] MongoDB connection failed: {e}. Falling back to in-memory storage. ⚠️")


# ── Load Model Once at Startup ──────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/data/mediscan_model.pt")


    # ── Auto Download Model if not exists ──────────────────
def ensure_model_exists():
    if not Path(MODEL_PATH).exists():
        print("[api] Model not found locally — checking HF Space files...")
        # Model HF Space files mein hai
        hf_model = Path("/app/mediscan_model.pt")
        if hf_model.exists():
            print("[api] Model found in /app ✅")
        else:
            print("[api] Model not found ⚠️")

ensure_model_exists()


def generate_gradcam(image: Image.Image, tensor: torch.Tensor) -> str:
    """
    Generate Grad-CAM heatmap and return as base64 string.
# ── Helper: Generate Grad-CAM ───────────────────────────
    """
    if model is None:
        return ""

    try:
        target_layer = model.layer4[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])

        grayscale_cam = cam(
            input_tensor=tensor.to(DEVICE),
            targets=None,
        )[0]

        img_resized = np.array(
            image.convert("RGB").resize((224, 224))
        ) / 255.0

        cam_image = show_cam_on_image(
            img_resized.astype(np.float32),
            grayscale_cam,
            use_rgb=True,
        )

        _, buffer = cv2.imencode(".jpg", cam_image)
        encoded   = base64.b64encode(buffer).decode("utf-8")
        return encoded

    except Exception as e:
        print(f"[api] Grad-CAM error: {e}")
        return ""


# ══════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the MediScan dashboard."""
    with open("app.html", "r") as f:
        return f.read()
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_loaded":  model is not None,
        "model_path":    MODEL_PATH,
        "version":       "0.1.0",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


# ── POST /predict ─────────────────────────────────────── (UPDATED)
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept a chest X-ray image.
    Supports both normal file upload and base64 data URL.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first.",
        )

    # Validate file type
    if file.content_type and file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are accepted.",
        )

    # Read contents
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # === IMPROVED IMAGE LOADING ===
    image = None
    try:
        # Try as normal binary image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"[api] ✅ Normal image received: {file.filename}")
        
    except (UnidentifiedImageError, Exception):
        # Try as base64 (common when frontend sends data URL)
        try:
            data_str = contents.decode('utf-8').strip()
            # Remove data URL prefix if exists
            if data_str.startswith('data:'):
                data_str = data_str.split(',')[-1]
            
            image_data = base64.b64decode(data_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            print("[api] ✅ Base64 image received")
            
        except Exception as e:
            print(f"[api] ❌ Image decode failed: {e}")
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload a valid JPEG or PNG image."
            )

    # Image hash
    img_hash = hashlib.sha256(contents).hexdigest()

    # Preprocess & Predict
    tensor = preprocess_single_image(image)
    result = predict(model, tensor)

    # Grad-CAM
    gradcam_b64 = generate_gradcam(image, tensor)

    # Response
    response = {
        "image_hash":      img_hash,
        "predicted_class": result["predicted_class"],
        "probability":     result["probability"],
        "risk_level":      result["risk_level"],
        "recommendation":  result["recommendation"],
        "gradcam_heatmap": gradcam_b64,
        "model_version":   "0.1.0",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }

    # Log prediction
    log_doc = {**response, "filename": file.filename}
    if mongo_available:
        try:
            predictions_col.insert_one(log_doc)
            print(f"[api] Prediction logged to MongoDB — {result['predicted_class']}")
        except Exception as e:
            print(f"[api] MongoDB log error: {e}. Logging in-memory instead.")
            in_memory_predictions.insert(0, log_doc)
    else:
        in_memory_predictions.insert(0, log_doc)
        print(f"[api] Prediction logged in-memory — {result['predicted_class']}")

    return response


# ── GET /history ─────────────────────────────────────────
@app.get("/history")
def history():
    if mongo_available:
        try:
            docs = list(
                predictions_col.find(
                    {}, 
                    {"_id": 0, "gradcam_heatmap": 0}
                ).sort("timestamp", -1).limit(20)
            )
            return {"count": len(docs), "predictions": docs}
        except Exception as e:
            print(f"[api] MongoDB history error: {e}. Using in-memory fallback.")
    
    # In-memory fallback: return list without heatmaps
    docs = []
    for item in in_memory_predictions[:20]:
        doc_copy = item.copy()
        doc_copy.pop("gradcam_heatmap", None)
        docs.append(doc_copy)
    return {"count": len(docs), "predictions": docs}


# ── GET /stats ───────────────────────────────────────────
@app.get("/stats")
def stats():
    if mongo_available:
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_predictions": {"$sum": 1},
                        "high_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "HIGH"]}, 1, 0]}},
                        "medium_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "MEDIUM"]}, 1, 0]}},
                        "low_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "LOW"]}, 1, 0]}},
                        "avg_probability": {"$avg": "$probability"},
                        "pneumonia_count": {"$sum": {"$cond": [{"$eq": ["$predicted_class", "PNEUMONIA"]}, 1, 0]}},
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "total_predictions": 1,
                        "high_risk_count": 1,
                        "medium_risk_count": 1,
                        "low_risk_count": 1,
                        "pneumonia_count": 1,
                        "avg_probability": {"$round": ["$avg_probability", 4]},
                    }
                },
            ]

            result = list(predictions_col.aggregate(pipeline))
            if result:
                return result[0]
        except Exception as e:
            print(f"[api] MongoDB stats error: {e}. Using in-memory fallback.")

    # Calculate in-memory
    total = len(in_memory_predictions)
    if total == 0:
        return {
            "total_predictions": 0, "high_risk_count": 0,
            "medium_risk_count": 0, "low_risk_count": 0,
            "pneumonia_count": 0, "avg_probability": 0.0
        }
    
    high_risk = sum(1 for p in in_memory_predictions if p["risk_level"] == "HIGH")
    medium_risk = sum(1 for p in in_memory_predictions if p["risk_level"] == "MEDIUM")
    low_risk = sum(1 for p in in_memory_predictions if p["risk_level"] == "LOW")
    pneumonia = sum(1 for p in in_memory_predictions if p["predicted_class"] == "PNEUMONIA")
    avg_prob = sum(p["probability"] for p in in_memory_predictions) / total
    
    return {
        "total_predictions": total,
        "high_risk_count": high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count": low_risk,
        "pneumonia_count": pneumonia,
        "avg_probability": round(avg_prob, 4)
    }