from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from .model import PlantDiseaseModel, load_image

app = FastAPI(title="Plant Disease API")

# Chargé une seule fois au démarrage de l'API
model = PlantDiseaseModel("../models/model.onnx", "../models/classes.json")


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