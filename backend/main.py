from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io, os, urllib.request, json
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, Input, BatchNormalization, Activation, Add, ZeroPadding2D, AveragePooling2D, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

app = FastAPI(title="MediScan AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HF_BASE = "https://huggingface.co/RadiologyApp/mediscan-models/resolve/main"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
MODELS = {}

def download_file(filename):
    local_path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(local_path):
        print(f"⬇️  Downloading {filename}...")
        urllib.request.urlretrieve(f"{HF_BASE}/{filename}", local_path)
        print(f"✅ Downloaded {filename}")
    return local_path

def strip_quantization(config):
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        for v in config.values():
            strip_quantization(v)
    elif isinstance(config, list):
        for item in config:
            strip_quantization(item)
    return config

def load_model_from_json_and_weights(config_file, weights_file):
    config_path = download_file(config_file)
    weights_path = download_file(weights_file)
    with open(config_path, "r") as f:
        config = json.load(f)
    config = strip_quantization(config)
    model = tf.keras.models.model_from_json(json.dumps(config))
    model.load_weights(weights_path)
    return model

def load_models():
    try:
        MODELS["pneumonia"] = load_model_from_json_and_weights(
            "pneumonia_config.json", "pneumonia_weights.weights.h5")
        print("✅ Loaded: pneumonia")
    except Exception as e:
        print(f"❌ Failed pneumonia: {e}")
    try:
        MODELS["skin_cancer"] = load_model_from_json_and_weights(
            "skin_cancer_config.json", "skin_cancer_weights.weights.h5")
        print("✅ Loaded: skin_cancer")
    except Exception as e:
        print(f"❌ Failed skin_cancer: {e}")
    try:
        MODELS["brain_tumor"] = load_model_from_json_and_weights(
            "brain_tumor_config.json", "brain_tumor_weights.weights.h5")
        print("✅ Loaded: brain_tumor")
    except Exception as e:
        print(f"❌ Failed brain_tumor: {e}")

@app.on_event("startup")
def startup_event():
    load_models()

PNEUMONIA_LABELS = {0: "Normal", 1: "Pneumonia"}
SKIN_LABELS = {0: "Actinic Keratoses", 1: "Basal Cell Carcinoma", 2: "Benign Keratosis", 3: "Dermatofibroma", 4: "Melanoma", 5: "Melanocytic Nevus", 6: "Vascular Lesion"}
BRAIN_LABELS = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"}

def preprocess_pneumonia(img):
    img = img.convert("RGB").resize((128, 128))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

def preprocess_skin(img):
    img = img.convert("RGB").resize((299, 299))
    return np.expand_dims(inception_preprocess(np.array(img, dtype=np.float32)), axis=0)

def preprocess_brain(img):
    img = img.convert("RGB").resize((224, 224))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

@app.get("/")
def root():
    return {"status": "MediScan AI is running", "models_loaded": list(MODELS.keys())}

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    if model_name not in ["pneumonia", "skin_cancer", "brain_tumor"]:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    if model_name not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except:
        raise HTTPException(status_code=400, detail="Could not open image.")
    if model_name == "pneumonia":
        arr = preprocess_pneumonia(img)
    elif model_name == "skin_cancer":
        arr = preprocess_skin(img)
    else:
        arr = preprocess_brain(img)
    raw = MODELS[model_name].predict(arr)
    if model_name == "pneumonia":
        prob = float(raw[0][0])
        label_idx = 1 if prob > 0.5 else 0
        label = PNEUMONIA_LABELS[label_idx]
        confidence = prob if label_idx == 1 else 1 - prob
        all_probs = {"Normal": round((1-prob)*100,1), "Pneumonia": round(prob*100,1)}
    elif model_name == "skin_cancer":
        probs = raw[0]
        label_idx = int(np.argmax(probs))
        label = SKIN_LABELS[label_idx]
        confidence = float(probs[label_idx])
        all_probs = {SKIN_LABELS[i]: round(float(p)*100,1) for i,p in enumerate(probs)}
    else:
        probs = raw[0]
        label_idx = int(np.argmax(probs))
        label = BRAIN_LABELS[label_idx]
        confidence = float(probs[label_idx])
        all_probs = {BRAIN_LABELS[i]: round(float(p)*100,1) for i,p in enumerate(probs)}
    return {"model": model_name, "prediction": label, "confidence": round(confidence*100,1), "all_probabilities": all_probs}
