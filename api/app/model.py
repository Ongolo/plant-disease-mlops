import io
import json

import numpy as np
import onnxruntime as ort
from PIL import Image

# Normalisation standard ImageNet (la même qui servira à l'entraînement)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(data: bytes) -> Image.Image:
    """Transforme les octets reçus en image RGB."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    """Image -> tableau [1, 3, 224, 224] attendu par le modèle."""
    img = img.resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0   # pixels entre 0 et 1
    x = (x - MEAN) / STD                            # normalisation
    x = x.transpose(2, 0, 1)                        # [224,224,3] -> [3,224,224]
    return x[None, ...]                             # ajoute la dimension batch


class PlantDiseaseModel:
    def __init__(self, model_path: str, classes_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        with open(classes_path) as f:
            self.classes = json.load(f)

    def predict(self, img: Image.Image, top_k: int = 3) -> list[dict]:
        logits = self.session.run(None, {self.input_name: preprocess(img)})[0][0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()                  # softmax -> probabilités
        best = np.argsort(probs)[::-1][:top_k]       # les k meilleures classes
        return [{"label": self.classes[i], "confidence": float(probs[i])} for i in best]