import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_TOKEN = os.getenv("UI_TOKEN", "")

st.set_page_config(page_title="Diagnostic des plantes", page_icon="🌿")
st.title("🌿 Diagnostic de maladies des plantes")
st.caption("Deposez la photo d'une feuille : le modele predit la maladie parmi 38 classes.")

# Token envoye a l'API (pre-rempli avec celui de l'interface)
token = st.sidebar.text_input("Token API", value=DEFAULT_TOKEN, type="password")
headers = {"Authorization": f"Bearer {token}"}

file = st.file_uploader("Photo de feuille", type=["jpg", "jpeg", "png"])

if file:
    st.image(file, width=300)

    if st.button("Diagnostiquer", type="primary"):
        try:
            r = requests.post(f"{API_URL}/predict", headers=headers,
                              files={"file": (file.name, file.getvalue(), file.type)},
                              timeout=30)
        except requests.ConnectionError:
            st.error(f"API injoignable a l'adresse {API_URL}")
            st.stop()

        if r.status_code == 401:
            st.error("Acces refuse : token manquant ou invalide.")
        elif r.ok:
            res = r.json()
            plant, _, disease = res["prediction"]["label"].partition("___")
            st.success(f"**{plant}** — {disease.replace('_', ' ')}")
            st.metric("Confiance", f"{res['prediction']['confidence']:.1%}")
            st.write("Top 3 :")
            st.table(res["top_k"])
            st.caption(f"Temps de prediction : {res['latency_ms']} ms")
        else:
            st.error(f"Erreur {r.status_code} : {r.text}")

# Suivi d'usage de l'utilisateur courant
if st.sidebar.button("Mon usage"):
    r = requests.get(f"{API_URL}/usage/me", headers=headers, timeout=10)
    if r.ok:
        st.sidebar.json(r.json())
    else:
        st.sidebar.error(f"Erreur {r.status_code}")