import io
import json

import numpy as np
import onnxruntime as ort
from PIL import Image

# Normalisation ImageNet (identique a l'entrainement)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(data: bytes) -> Image.Image:
    """Transforme les octets recus en image RGB."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    """Image -> tableau [1, 3, 224, 224] attendu par le modele."""
    img = img.resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)
    return x[None, ...]


def image_stats(img: Image.Image) -> dict:
    """Quelques statistiques simples de l'image (pour le data drift)."""
    arr = np.asarray(img, dtype=np.float32)
    gray = arr.mean(axis=2)
    return {
        "width": img.width,
        "height": img.height,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "mean_r": float(arr[..., 0].mean()),
        "mean_g": float(arr[..., 1].mean()),
        "mean_b": float(arr[..., 2].mean()),
    }


class PlantDiseaseModel:
    def __init__(self, model_path: str, classes_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        with open(classes_path) as f:
            self.classes = json.load(f)

    def predict(self, img: Image.Image, top_k: int = 3) -> list[dict]:
        logits = self.session.run(None, {self.input_name: preprocess(img)})[0][0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        best = np.argsort(probs)[::-1][:top_k]
        return [{"label": self.classes[i], "confidence": float(probs[i])} for i in best]