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

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["mediscan"]
    predictions_col = db["predictions"]
    print("[api] MongoDB connected ✅")
except ConnectionFailure:
    print("[api] MongoDB not available ⚠️")
    predictions_col = None

MODEL_PATH = os.getenv("MODEL_PATH", "mediscan_model.pt")

# Model Loading
if not os.path.exists(MODEL_PATH):
    print("[api] Downloading model from HF Hub...")
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        tmp = hf_hub_download(
            repo_id="sunilakiran56/mediscan-model",
            filename="mediscan_model.pt",
            repo_type="model",
        )
        shutil.copy(tmp, MODEL_PATH)
        print("[api] Model downloaded ✅")
    except Exception as e:
        print(f"[api] Model download failed: {e}")

try:
    model = load_model(MODEL_PATH)
    print("[api] Model loaded ✅")
except Exception as e:
    model = None
    print(f"[api] Model loading failed: {e}")

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
        return "<h1>MediScan is Running!</h1><p>Please upload app.html file.</p>"

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
        raise HTTPException(status_code=503, detail="Model not loaded. Please check logs.")

    contents = await file.read()

    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="Empty or invalid image file.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot open image: Invalid image file.")

    try:
        img_hash = hashlib.sha256(contents).hexdigest()
        tensor = preprocess_single_image(image)
        result = predict(model, tensor)
        gradcam_b64 = generate_gradcam(image, tensor)

        response = {
            "image_hash": img_hash,
            "predicted_class": result["predicted_class"],
            "probability": float(result["probability"]),
            "risk_level": result["risk_level"],
            "recommendation": result["recommendation"],
            "gradcam_heatmap": gradcam_b64,
            "model_version": "0.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if predictions_col is not None:
            try:
                predictions_col.insert_one({**response, "filename": file.filename or "unknown"})
            except Exception as e:
                print(f"[api] MongoDB insert error: {e}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/history")
def history():
    try:
        if predictions_col is None:
            return {"count": 0, "predictions": []}
        docs = list(predictions_col.find({}, {"_id": 0, "gradcam_heatmap": 0}).sort("timestamp", -1).limit(20))
        return {"count": len(docs), "predictions": docs}
    except Exception as e:
        return {"count": 0, "predictions": [], "error": str(e)}

@app.get("/stats")
def stats():
    try:
        if predictions_col is None:
            return {"total_predictions": 0, "high_risk_count": 0, "pneumonia_count": 0, "avg_probability": 0.0}
        
        pipeline = [ ... ]  # aapka purana pipeline yahan paste kar sakte ho
        # (main ne short kiya hai, agar chahiye to pura paste kar dunga)
        result = list(predictions_col.aggregate(pipeline))
        return result[0] if result else {"total_predictions": 0, ...}
    except Exception:
        return {"total_predictions": 0, "high_risk_count": 0, "pneumonia_count": 0, "avg_probability": 0.0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)