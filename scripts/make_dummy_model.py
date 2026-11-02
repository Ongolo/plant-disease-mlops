"""Génère un modèle ONNX FACTICE pour tester l'infrastructure.
Entrée  : image [N, 3, 224, 224]  ->  Sortie : 38 scores [N, 38]
"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight",
    "Corn___healthy", "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight",
    "Grape___healthy", "Orange___Citrus_greening", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites",
    "Tomato___Target_Spot", "Tomato___Yellow_Leaf_Curl_Virus", "Tomato___Mosaic_virus",
    "Tomato___healthy",
]

# Poids aléatoires : 3 canaux de couleur -> 38 classes
rng = np.random.default_rng(0)
W = numpy_helper.from_array(rng.normal(size=(3, 38)).astype(np.float32), "W")
B = numpy_helper.from_array(np.zeros(38, dtype=np.float32), "B")

# Le "réseau" : moyenne de chaque canal de couleur, puis une couche linéaire
graph = helper.make_graph(
    nodes=[
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"]),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logits"]),
    ],
    name="dummy_plant_classifier",
    inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 224, 224])],
    outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 38])],
    initializer=[W, B],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)

out = Path("models")
out.mkdir(exist_ok=True)
onnx.save(model, out / "model.onnx")
(out / "classes.json").write_text(json.dumps(CLASSES, indent=2))
print("Modèle factice créé :", out / "model.onnx")