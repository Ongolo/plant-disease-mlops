import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_TOKEN = os.getenv("UI_TOKEN", "")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/reports"))   # volume partage avec le service monitoring
LOW_CONFIDENCE = 0.7

st.set_page_config(page_title="Diagnostic des plantes", page_icon="🌿", layout="wide")


@st.cache_data(ttl=30)
def api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


# ---------- Barre laterale : connexion et usage ----------
st.sidebar.header("Connexion")
token = st.sidebar.text_input("Token API", value=DEFAULT_TOKEN, type="password")
headers = {"Authorization": f"Bearer {token}"}

health = api_health()
if health:
    st.sidebar.success("API en ligne")
    st.sidebar.caption(f"Modele en production : **{health['model_version']}**")
else:
    st.sidebar.error(f"API injoignable ({API_URL})")

if st.sidebar.button("Mon usage"):
    r = requests.get(f"{API_URL}/usage/me", headers=headers, timeout=10)
    if r.ok:
        u = r.json()
        st.sidebar.metric("Requetes", u["n_requests"])
        st.sidebar.metric("Temps moyen du modele", f"{u['avg_latency_ms']} ms")
        st.sidebar.caption(f"Utilisateur : {u['user']}")
    else:
        st.sidebar.error(f"Erreur {r.status_code}")

st.title("🌿 Diagnostic de maladies des plantes")
tab_diag, tab_monitor = st.tabs(["🔍 Diagnostic", "📈 Monitoring du modele"])

# ---------- Onglet 1 : diagnostic d'une photo ----------
with tab_diag:
    st.caption("Deposez la photo d'une feuille : le modele predit la maladie parmi 38 classes.")
    file = st.file_uploader("Photo de feuille", type=["jpg", "jpeg", "png"])

    if file:
        col_img, col_res = st.columns([1, 2])
        col_img.image(file, use_container_width=True)

        if col_res.button("Diagnostiquer", type="primary"):
            try:
                r = requests.post(f"{API_URL}/predict", headers=headers,
                                  files={"file": (file.name, file.getvalue(), file.type)},
                                  timeout=30)
            except requests.ConnectionError:
                col_res.error(f"API injoignable a l'adresse {API_URL}")
                st.stop()

            if r.status_code == 401:
                col_res.error("Acces refuse : token manquant ou invalide.")
            elif r.ok:
                res = r.json()
                conf = res["prediction"]["confidence"]
                plant, _, disease = res["prediction"]["label"].partition("___")
                disease = disease.replace("_", " ")
                if "healthy" in disease.lower():
                    col_res.success(f"### {plant} — feuille saine")
                else:
                    col_res.warning(f"### {plant} — {disease}")

                m1, m2, m3 = col_res.columns(3)
                m1.metric("Confiance", f"{conf:.1%}")
                m2.metric("Temps du modele", f"{res['latency_ms']:.0f} ms")
                m3.metric("Version", res.get("model_version", "?").split()[-1])

                if conf < LOW_CONFIDENCE:
                    col_res.error("Confiance faible : la photo ressemble peu aux images d'entrainement "
                                  "(photos de laboratoire). Resultat a verifier.")

                col_res.write("**Top 3 des hypotheses**")
                for p in res["top_k"]:
                    name = p["label"].replace("___", " — ").replace("_", " ")
                    col_res.progress(p["confidence"], text=f"{name} : {p['confidence']:.1%}")
            else:
                col_res.error(f"Erreur {r.status_code} : {r.text}")

# ---------- Onglet 2 : monitoring (data drift) ----------
with tab_monitor:
    summary_path = REPORTS_DIR / "drift_summary.json"
    if not summary_path.exists():
        st.info("Aucune analyse de drift disponible pour l'instant (le service monitoring "
                "calcule toutes les heures).")
    else:
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        st.caption(f"Derniere analyse : {s['timestamp']} — comparaison des "
                   f"{s.get('n_current', 0)} dernieres predictions aux images d'entrainement.")

        if s["status"] == "donnees insuffisantes":
            st.info(f"Donnees insuffisantes : {s['n_current']} predictions (minimum {s['min_rows']}).")
        else:
            if s["dataset_drift"]:
                st.error("🚨 **Data drift detecte** : les images recues ne ressemblent plus "
                         "aux images d'entrainement.")
            else:
                st.success("✅ **Pas de drift** : les images recues ressemblent aux images d'entrainement.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Variables en drift", f"{s['drifted_columns']} / {s['n_columns']}")
            c2.metric("Confiance moyenne du modele", f"{s['mean_confidence_current']:.3f}",
                      delta=f"{s['mean_confidence_current'] - s['mean_confidence_reference']:+.3f} "
                            f"vs entrainement")
            c3.metric("Predictions analysees", s["n_current"])

            pv = pd.DataFrame({"variable": list(s["p_values"]),
                               "p-value (test K-S)": list(s["p_values"].values())})
            pv["drift"] = pv["p-value (test K-S)"] < 0.05
            pv["drift"] = pv["drift"].map({True: "🚨 oui", False: "✅ non"})
            st.dataframe(pv, hide_index=True, use_container_width=True)

        # Historique des analyses
        hist_path = REPORTS_DIR / "drift_history.jsonl"
        if hist_path.exists():
            rows = [json.loads(line) for line in hist_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            hist = pd.DataFrame([r for r in rows if "mean_confidence_current" in r])
            if len(hist):
                hist["timestamp"] = pd.to_datetime(hist["timestamp"])
                st.write("**Historique : confiance moyenne du modele et part de variables en drift**")
                st.line_chart(hist.set_index("timestamp")[["mean_confidence_current", "drift_share"]])

        report_path = REPORTS_DIR / "drift_report.html"
        if report_path.exists():
            st.download_button("Telecharger le rapport Evidently complet (HTML)",
                               report_path.read_bytes(), file_name="drift_report.html",
                               mime="text/html")