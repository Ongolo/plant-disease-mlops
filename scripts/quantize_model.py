"""Quantization INT8 du modele ONNX + comparaison avec le modele FP32.

Usage :
  python scripts/quantize_model.py      (valeurs par defaut ci-dessous)

Produit dans --out : model.onnx (INT8), classes.json, metrics.json (taille, latence, accuracy).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXT = {".jpg", ".jpeg", ".png"}


def preprocess(path):  # identique a api/app/model.py
    img = Image.open(path).convert("RGB").resize((224, 224))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None, ...]


def images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXT)


class Calib(CalibrationDataReader):
    """Fournit quelques images reelles pour mesurer les plages de valeurs (calibration)."""
    def __init__(self, files, input_name):
        self.it = iter([{input_name: preprocess(f)} for f in files])

    def get_next(self):
        return next(self.it, None)


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1          # comme en production (1 thread par worker)
    return ort.InferenceSession(str(path), sess_options=opts)


def evaluate(path, files, classes):
    s = session(path)
    name = s.get_inputs()[0].name
    xs = [preprocess(f) for f in files]
    for x in xs[:5]:                       # echauffement
        s.run(None, {name: x})
    preds, t0 = [], time.perf_counter()
    for x in xs:
        preds.append(int(s.run(None, {name: x})[0][0].argmax()))
    latency = (time.perf_counter() - t0) / len(xs) * 1000
    # vraie classe = debut du nom de fichier (ex. "Apple___Apple_scab__xxx.jpg")
    norm = [c.replace(" ", "_") for c in classes]
    labels = [next((i for i, c in enumerate(norm) if f.name.startswith(c + "__")), None) for f in files]
    known = [(p, l) for p, l in zip(preds, labels) if l is not None]
    acc = sum(p == l for p, l in known) / len(known) if known else None
    return preds, latency, acc, len(known)


p = argparse.ArgumentParser()
p.add_argument("--model", default="models/model.onnx")
p.add_argument("--classes", default="models/classes.json")
p.add_argument("--calib", default="demo_images/plantvillage")
p.add_argument("--eval", default="demo_images/plantvillage")
p.add_argument("--n-calib", type=int, default=50)
p.add_argument("--out", default="models_int8")
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
prep = out / "model_prep.onnx"
quant_pre_process(args.model, str(prep))   # optimisation du graphe avant quantization

input_name = ort.InferenceSession(args.model).get_inputs()[0].name
calib_files = images(args.calib)[: args.n_calib]
quantize_static(str(prep), str(out / "model.onnx"), Calib(calib_files, input_name),
                quant_format=QuantFormat.QDQ, per_channel=True,
                calibrate_method=CalibrationMethod.Percentile,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
prep.unlink()
shutil.copy(args.classes, out / "classes.json")

classes = json.load(open(args.classes))
eval_files = [f for f in images(args.eval) if f not in calib_files]  # images jamais vues a la calibration
fp32_preds, fp32_ms, fp32_acc, n_lab = evaluate(args.model, eval_files, classes)
int8_preds, int8_ms, int8_acc, _ = evaluate(out / "model.onnx", eval_files, classes)
agree = float(np.mean(np.array(fp32_preds) == np.array(int8_preds)))
size = lambda f: Path(f).stat().st_size / 1e6

metrics = {
    "arch": "MobileNetV3-Large INT8 (QDQ, per-channel, calibration percentile)",
    "fp32": {"size_mb": round(size(args.model), 2), "latency_ms": round(fp32_ms, 2),
             "accuracy": None if fp32_acc is None else round(fp32_acc, 4)},
    "int8": {"size_mb": round(size(out / "model.onnx"), 2), "latency_ms": round(int8_ms, 2),
             "accuracy": None if int8_acc is None else round(int8_acc, 4)},
    "agreement_fp32_int8": round(agree, 4),
    "n_eval_images": len(eval_files), "n_labelled": n_lab, "n_calib": len(calib_files),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))