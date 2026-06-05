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

app = FastAPI(title="MediScan API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["mediscan"]
    predictions_col = db["predictions"]
    training_runs_col = db["training_runs"]
    print("[api] MongoDB connected ✅")
except ConnectionFailure:
    print("[api] MongoDB not available ⚠️")

MODEL_PATH = os.getenv("MODEL_PATH", "mediscan_model.pt")

if not os.path.exists(MODEL_PATH):
    print("[api] Downloading model from HF Hub...")
    try:
        import shutil
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(
            repo_id="sunilakiran56/mediscan-model",
            filename="mediscan_model.pt",
            repo_type="model",
        )
        shutil.copy(tmp, MODEL_PATH)
        print("[api] Model downloaded ✅")
    except Exception as e:
        print(f"[api] Download failed: {e}")

try:
    model = load_model(MODEL_PATH)
    print("[api] Model loaded ✅")
except Exception:
    model = None
    print("[api] Model not found ⚠️")


def generate_gradcam(image, tensor):
    if model is None:
        return ""
    try:
        target_layer = model.layer4[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=tensor.to(DEVICE), targets=None)[0]
        img_resized = np.array(image.convert("RGB").resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(img_resized.astype(np.float32), grayscale_cam, use_rgb=True)
        _, buffer = cv2.imencode(".jpg", cam_image)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"[api] Grad-CAM error: {e}")
        return ""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    try:
        with open("app.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>MediScan API Running!</h1><a href='/docs'>Docs</a>"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = await file.read()
    print(f"[api] Received file: {file.filename}, size: {len(contents)} bytes, type: {file.content_type}")

    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # Try multiple ways to open image
    image = None
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e1:
        print(f"[api] PIL error: {e1}")
        # Try saving to temp file first
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            image = Image.open(tmp_path).convert("RGB")
            os.unlink(tmp_path)
        except Exception as e2:
            print(f"[api] Temp file error: {e2}")
            raise HTTPException(status_code=400, detail=f"Cannot open image: {e1}")

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
        predictions_col.insert_one({**response, "filename": file.filename or "unknown"})
    except Exception as e:
        print(f"[api] MongoDB error: {e}")

    return response


@app.get("/history")
def history():
    try:
        docs = list(predictions_col.find({}, {"_id": 0, "gradcam_heatmap": 0}).sort("timestamp", -1).limit(20))
        return {"count": len(docs), "predictions": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    try:
        pipeline = [
            {"$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "HIGH"]}, 1, 0]}},
                "medium_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "MEDIUM"]}, 1, 0]}},
                "low_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "LOW"]}, 1, 0]}},
                "avg_probability": {"$avg": "$probability"},
                "pneumonia_count": {"$sum": {"$cond": [{"$eq": ["$predicted_class", "PNEUMONIA"]}, 1, 0]}},
            }},
            {"$project": {
                "_id": 0,
                "total_predictions": 1,
                "high_risk_count": 1,
                "medium_risk_count": 1,
                "low_risk_count": 1,
                "pneumonia_count": 1,
                "avg_probability": {"$round": ["$avg_probability", 4]},
            }},
        ]
        result = list(predictions_col.aggregate(pipeline))
        if not result:
            return {"total_predictions": 0, "high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 0, "pneumonia_count": 0, "avg_probability": 0.0}
        return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))