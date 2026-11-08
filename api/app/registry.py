import logging
import os
import time

log = logging.getLogger("uvicorn.error")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MODEL_NAME = os.getenv("MODEL_NAME", "plant-disease-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")


def fetch_production_model(attempts: int = 5, wait_s: int = 5):
    """Telecharge la version du modele qui porte l'alias 'production' dans MLflow.

    Renvoie (chemin_model_onnx, chemin_classes_json, libelle_version),
    ou None si MLflow n'est pas configure ou injoignable : l'API utilise alors les fichiers locaux.
    """
    if not MLFLOW_TRACKING_URI:
        return None

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for i in range(1, attempts + 1):
        try:
            mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            dest = os.path.join(CACHE_DIR, f"v{mv.version}-pid{os.getpid()}")  # 1 dossier par worker            os.makedirs(dest, exist_ok=True)
            path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}", dst_path=dest)
            label = f"{MODEL_NAME} v{mv.version}"
            log.info("Modele charge depuis MLflow : %s (alias %s)", label, MODEL_ALIAS)
            return os.path.join(path, "model.onnx"), os.path.join(path, "classes.json"), label
        except Exception as exc:  # MLflow pas encore pret, alias inexistant, reseau...
            log.warning("MLflow : tentative %d/%d echouee (%s)", i, attempts, str(exc)[:150])
            if i < attempts:
                time.sleep(wait_s)
    log.warning("MLflow indisponible : utilisation des fichiers locaux")
    return None