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

# ====================== MongoDB ======================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["mediscan"]
    predictions_col = db["predictions"]
    print("[api] MongoDB connected ✅")
except ConnectionFailure:
    print("[api] MongoDB not available ⚠️")
    predictions_col = None

# ====================== Model ======================
MODEL_PATH = os.getenv("MODEL_PATH", "mediscan_model.pt")

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

# ====================== Routes ======================
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    try:
        with open("app.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>MediScan API Running! Please upload app.html</h1>"

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = await file.read()
    file_size = len(contents)

    print(f"[api] Received file: {file.filename}, size: {file_size} bytes, type: {file.content_type}")

    # MacOS Hidden File Protection
    if file.filename.startswith("._") or "__MACOSX" in file.filename:
        raise HTTPException(status_code=400, detail="Invalid macOS hidden file. Please upload the actual image.")

    if file_size < 1024:
        raise HTTPException(status_code=400, detail=f"File too small ({file_size} bytes). Not a valid image.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"[api] Image opened successfully: {image.size}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {str(e)}")

    # Prediction
    tensor = preprocess_single_image(image)
    result = predict(model, tensor)
    gradcam_b64 = generate_gradcam(image, tensor)

    response = {
        "predicted_class": result["predicted_class"],
        "probability": float(result["probability"]),
        "risk_level": result["risk_level"],
        "recommendation": result["recommendation"],
        "gradcam_heatmap": gradcam_b64,
        "model_version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save to MongoDB
    if predictions_col is not None:
        try:
            predictions_col.insert_one({**response, "filename": file.filename or "unknown"})
        except Exception as e:
            print(f"[api] MongoDB error: {e}")

    return response


@app.get("/stats")
def stats():
    try:
        if predictions_col is None:
            return {"total_predictions": 0, "high_risk_count": 0, "pneumonia_count": 0, "avg_probability": 0.0}
        
        pipeline = [
            {"$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "HIGH"]}, 1, 0]}},
                "pneumonia_count": {"$sum": {"$cond": [{"$eq": ["$predicted_class", "PNEUMONIA"]}, 1, 0]}},
                "avg_probability": {"$avg": "$probability"},
            }},
            {"$project": {
                "_id": 0,
                "total_predictions": 1,
                "high_risk_count": 1,
                "pneumonia_count": 1,
                "avg_probability": {"$round": ["$avg_probability", 4]},
            }}
        ]
        result = list(predictions_col.aggregate(pipeline))
        return result[0] if result else {"total_predictions": 0, "high_risk_count": 0, "pneumonia_count": 0, "avg_probability": 0.0}
    except Exception as e:
        print(f"[api] Stats error: {e}")
        return {"total_predictions": 0, "high_risk_count": 0, "pneumonia_count": 0, "avg_probability": 0.0}


@app.get("/history")
def history():
    try:
        if predictions_col is None:
            return {"predictions": []}
        docs = list(predictions_col.find({}, {"_id": 0, "gradcam_heatmap": 0}).sort("timestamp", -1).limit(15))
        return {"predictions": docs}
    except Exception as e:
        print(f"[api] History error: {e}")
        return {"predictions": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)