import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import UnidentifiedImageError
from sqlalchemy import func
from sqlalchemy.orm import Session

from .db import PredictionLog, User, get_db, init_db
from .model import PlantDiseaseModel, image_stats, load_image

MODEL_PATH = os.getenv("MODEL_PATH", "../models/model.onnx")
CLASSES_PATH = os.getenv("CLASSES_PATH", "../models/classes.json")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Au demarrage : creation des tables + utilisateurs, chargement du modele
    init_db()
    state["model"] = PlantDiseaseModel(MODEL_PATH, CLASSES_PATH)
    yield


app = FastAPI(title="Plant Disease API", lifespan=lifespan)
bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                 db: Session = Depends(get_db)) -> User:
    """Verifie le header 'Authorization: Bearer <token>' et renvoie l'utilisateur."""
    if creds is None:
        raise HTTPException(status_code=401, detail="Token manquant (header Authorization: Bearer <token>)")
    user = db.query(User).filter_by(token=creds.credentials).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Token invalide")
    return user


@app.get("/health")
def health():
    # Reste public : Docker l'utilise pour verifier que l'API est vivante
    return {"status": "ok", "n_classes": len(state["model"].classes)}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    data = await file.read()
    try:
        img = load_image(data)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Fichier image invalide")

    t0 = time.perf_counter()
    preds = state["model"].predict(img)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Enregistrement de la requete dans PostgreSQL
    db.add(PredictionLog(user_id=user.id,
                         predicted_class=preds[0]["label"],
                         confidence=preds[0]["confidence"],
                         latency_ms=latency_ms,
                         **image_stats(img)))
    db.commit()

    return {"prediction": preds[0], "top_k": preds, "latency_ms": round(latency_ms, 2)}


@app.get("/usage/me")
def my_usage(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Nombre de requetes et latence moyenne de l'utilisateur qui appelle."""
    n, avg_latency, last = (
        db.query(func.count(PredictionLog.id),
                 func.avg(PredictionLog.latency_ms),
                 func.max(PredictionLog.created_at))
        .filter(PredictionLog.user_id == user.id)
        .one()
    )
    return {"user": user.name, "n_requests": n,
            "avg_latency_ms": round(avg_latency or 0, 2), "last_request": last}