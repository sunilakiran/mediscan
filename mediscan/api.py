"""
api.py
FastAPI application for MediScan.
"""

import io
import os
import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dotenv import load_dotenv

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

# ── MongoDB Setup ───────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "mediscan"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    predictions_col = db["predictions"]
    training_runs_col = db["training_runs"]
    print("[api] MongoDB connected ✅")
except ConnectionFailure:
    print("[api] MongoDB not available ⚠️")

# ── Model Path ──────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "mediscan_model.pt")


# ── Auto Download Model ─────────────────────────────────
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("[api] Downloading model from HF Hub...")
        try:
            import shutil
            from huggingface_hub import hf_hub_download
            tmp_path = hf_hub_download(
                repo_id="sunilakiran56/mediscan-model",
                filename="mediscan_model.pt",
                repo_type="model",
            )
            shutil.copy(tmp_path, MODEL_PATH)
            print("[api] Model downloaded ✅")
        except Exception as e:

            print(f"[api] Model download failed: {e}")
# ── Load Model ──────────────────────────────────────────
try:
    model = load_model(MODEL_PATH)
    print("[api] Model loaded ✅")
except FileNotFoundError:
    model = None
    print("[api] Model not found ⚠️")


# ── Grad-CAM ────────────────────────────────────────────
def generate_gradcam(image: Image.Image, tensor: torch.Tensor) -> str:
    if model is None:
        return ""
    try:
        target_layer = model.layer4[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=tensor.to(DEVICE), targets=None)[0]
        img_resized = np.array(image.convert("RGB").resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(
            img_resized.astype(np.float32),
            grayscale_cam,
            use_rgb=True,
        )
        _, buffer = cv2.imencode(".jpg", cam_image)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"[api] Grad-CAM error: {e}")
        return ""


# ══════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    try:
        with open("app.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>MediScan API Running!</h1><p><a href='/docs'>API Docs</a></p>"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
#Accept all uploads — validate by trying to open
    

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_hash = hashlib.sha256(contents).hexdigest()
    tensor = preprocess_single_image(image)
    result = predict(model, tensor)
    gradcam_b64 = generate_gradcam(image, tensor)

    response = {
        "image_hash": img_hash,
        "predicted_class": result["predicted_class"],
        "probability": result["probability"],
        "risk_level": result["risk_level"],
        "recommendation": result["recommendation"],
        "gradcam_heatmap": gradcam_b64,
        "model_version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        predictions_col.insert_one({**response, "filename": file.filename})
    except Exception as e:
        print(f"[api] MongoDB log error: {e}")

    return response


@app.get("/history")
def history():
    try:
        docs = list(
            predictions_col.find(
                {}, {"_id": 0, "gradcam_heatmap": 0}
            ).sort("timestamp", -1).limit(20)
        )
        return {"count": len(docs), "predictions": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    try:
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_predictions": {"$sum": 1},
                    "high_risk_count": {
                        "$sum": {"$cond": [{"$eq": ["$risk_level", "HIGH"]}, 1, 0]}
                    },
                    "medium_risk_count": {
                        "$sum": {"$cond": [{"$eq": ["$risk_level", "MEDIUM"]}, 1, 0]}
                    },
                    "low_risk_count": {
                        "$sum": {"$cond": [{"$eq": ["$risk_level", "LOW"]}, 1, 0]}
                    },
                    "avg_probability": {"$avg": "$probability"},
                    "pneumonia_count": {
                        "$sum": {"$cond": [{"$eq": ["$predicted_class", "PNEUMONIA"]}, 1, 0]}
                    },
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

        if not result:
            return {
                "total_predictions": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "pneumonia_count": 0,
                "avg_probability": 0.0,
            }

        return result[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))