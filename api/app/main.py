import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from .model import PlantDiseaseModel, load_image

# Chemins lus dans les variables d'environnement (avec une valeur par défaut pour le local)
MODEL_PATH = os.getenv("MODEL_PATH", "../models/model.onnx")
CLASSES_PATH = os.getenv("CLASSES_PATH", "../models/classes.json")

app = FastAPI(title="Plant Disease API")
model = PlantDiseaseModel(MODEL_PATH, CLASSES_PATH)
@app.get("/health")
def health():
    return {"status": "ok", "n_classes": len(model.classes)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = load_image(data)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Fichier image invalide")

    preds = model.predict(img)
    return {"prediction": preds[0], "top_k": preds}