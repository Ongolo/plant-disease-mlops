"""Enregistre un modele ONNX dans le registre MLflow et (option) le promeut en production.

Usage (depuis la racine du projet) :
  python scripts/register_model.py --tracking-uri http://35.181.112.52:5000 \
      --model models/model.onnx --classes models/classes.json \
      --metrics training/metrics.json --description "MobileNetV3 FP32" --promote
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

p = argparse.ArgumentParser()
p.add_argument("--tracking-uri", required=True)
p.add_argument("--model", default="models/model.onnx")
p.add_argument("--classes", default="models/classes.json")
p.add_argument("--metrics", default="training/metrics.json")
p.add_argument("--name", default="plant-disease-classifier")
p.add_argument("--experiment", default="plant-disease")
p.add_argument("--description", default="")
p.add_argument("--promote", action="store_true", help="donner l'alias 'production' a cette version")
args = p.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
mlflow.set_experiment(args.experiment)
client = MlflowClient()

metrics = json.loads(Path(args.metrics).read_text()) if Path(args.metrics).exists() else {}

with mlflow.start_run(run_name=args.description or "register") as run:
    # 1. Parametres et metriques de l'entrainement
    for key in ("arch", "pretrained", "epochs", "batch_size", "lr", "n_train", "n_test"):
        if key in metrics:
            mlflow.log_param(key, metrics[key])
    for key in ("test_accuracy", "f1_macro"):
        if key in metrics:
            mlflow.log_metric(key, metrics[key])
    for h in metrics.get("history", []):
        mlflow.log_metric("epoch_test_accuracy", h["test_accuracy"], step=h["epoch"])
        mlflow.log_metric("epoch_train_loss", h["train_loss"], step=h["epoch"])
    mlflow.log_metric("model_size_mb", Path(args.model).stat().st_size / 1e6)

    # 2. Fichiers du modele, ranges dans le dossier d'artefacts "model/"
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(args.model, Path(tmp) / "model.onnx")
        shutil.copy(args.classes, Path(tmp) / "classes.json")
        mlflow.log_artifacts(tmp, artifact_path="model")

# 3. Nouvelle version dans le registre
mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", args.name)
if args.description:
    client.update_model_version(args.name, mv.version, description=args.description)
print(f"Modele '{args.name}' version {mv.version} enregistre (run {run.info.run_id})")

# 4. Promotion en production (l'API charge la version qui porte cet alias)
if args.promote:
    client.set_registered_model_alias(args.name, "production", mv.version)
    print(f"Alias 'production' -> version {mv.version}")