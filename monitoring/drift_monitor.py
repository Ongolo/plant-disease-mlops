"""Detection de data drift : compare les images recues en production aux images d'entrainement.

- Reference : monitoring/reference.csv (statistiques d'images PlantVillage du jeu de test)
- Production : les N dernieres lignes de prediction_logs (PostgreSQL)
- Sorties (dossier REPORTS_DIR) :
    drift_report.html   rapport Evidently complet
    drift_summary.json  resume du dernier calcul (lu par l'interface)
    drift_history.jsonl un resume par calcul (historique)

Usage : python drift_monitor.py --once      (un seul calcul)
        python drift_monitor.py             (boucle, toutes les INTERVAL_MIN minutes)
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./usage.db")
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "reference.csv")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))
WINDOW = int(os.getenv("DRIFT_WINDOW", "200"))          # nb de predictions recentes analysees
MIN_ROWS = int(os.getenv("DRIFT_MIN_ROWS", "30"))       # minimum pour lancer un calcul
INTERVAL_MIN = int(os.getenv("DRIFT_INTERVAL_MIN", "60"))
EXCLUDE_USERS = [u.strip() for u in os.getenv("DRIFT_EXCLUDE_USERS", "bob").split(",") if u.strip()]
DRIFT_SHARE = float(os.getenv("DRIFT_SHARE", "0.5"))    # part de colonnes en drift => alerte

FEATURES = ["brightness", "contrast", "mean_r", "mean_g", "mean_b", "confidence"]


def load_current() -> pd.DataFrame:
    """Dernieres predictions de production (hors utilisateurs exclus, ex. test de charge)."""
    excluded = ", ".join(f"'{u}'" for u in EXCLUDE_USERS) or "''"
    query = text(f"""
        SELECT p.created_at, p.model_version, {", ".join("p." + c for c in FEATURES)}
        FROM prediction_logs p JOIN users u ON u.id = p.user_id
        WHERE u.name NOT IN ({excluded}) AND p.brightness IS NOT NULL
        ORDER BY p.id DESC LIMIT :n
    """)
    with create_engine(DATABASE_URL).connect() as conn:
        return pd.read_sql(query, conn, params={"n": WINDOW})


def run_once() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE_PATH)[FEATURES]
    current = load_current()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if len(current) < MIN_ROWS:
        summary = {"timestamp": now, "status": "donnees insuffisantes",
                   "n_current": len(current), "min_rows": MIN_ROWS}
    else:
        evaluation = Report([DataDriftPreset(drift_share=DRIFT_SHARE)]).run(
            current_data=current[FEATURES], reference_data=reference)
        evaluation.save_html(str(REPORTS_DIR / "drift_report.html"))

        metrics = evaluation.dict()["metrics"]
        drifted = next(m["value"] for m in metrics if m["metric_name"].startswith("DriftedColumnsCount"))
        p_values = {m["config"]["column"]: float(m["value"])
                    for m in metrics if m["metric_name"].startswith("ValueDrift")}
        summary = {
            "timestamp": now,
            "status": "drift detecte" if drifted["share"] >= DRIFT_SHARE else "pas de drift",
            "dataset_drift": drifted["share"] >= DRIFT_SHARE,
            "drifted_columns": int(drifted["count"]),
            "n_columns": len(FEATURES),
            "drift_share": round(float(drifted["share"]), 3),
            "p_values": {k: round(v, 6) for k, v in p_values.items()},
            "columns_in_drift": [k for k, v in p_values.items() if v < 0.05],
            "n_current": len(current),
            "n_reference": len(reference),
            "period": [str(current["created_at"].min()), str(current["created_at"].max())],
            "model_versions": current["model_version"].fillna("inconnue").value_counts().to_dict(),
            "mean_confidence_current": round(float(current["confidence"].mean()), 4),
            "mean_confidence_reference": round(float(reference["confidence"].mean()), 4),
        }

    (REPORTS_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(REPORTS_DIR / "drift_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="un seul calcul puis arret")
    args = parser.parse_args()
    if args.once:
        run_once()
    else:
        while True:
            try:
                run_once()
            except Exception as exc:  # base indisponible, etc. : on reessaie au prochain cycle
                print(f"Erreur pendant le calcul de drift : {exc}", flush=True)
            time.sleep(INTERVAL_MIN * 60)