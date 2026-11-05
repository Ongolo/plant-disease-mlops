import os

import requests
import streamlit as st

# Adresse de l'API : localhost en local, "api" dans Docker Compose (étape 7)
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Diagnostic des plantes", page_icon="🌿")
st.title("🌿 Diagnostic de maladies des plantes")
st.caption("Déposez la photo d'une feuille : le modèle prédit la maladie parmi 38 classes.")

file = st.file_uploader("Photo de feuille", type=["jpg", "jpeg", "png"])

if file:
    st.image(file, width=300)

    if st.button("Diagnostiquer", type="primary"):
        try:
            r = requests.post(
                f"{API_URL}/predict",
                files={"file": (file.name, file.getvalue(), file.type)},
                timeout=30,
            )
        except requests.ConnectionError:
            st.error(f"API injoignable à l'adresse {API_URL}")
            st.stop()

        if r.ok:
            res = r.json()
            # "Tomato___Late_blight" -> plante "Tomato", maladie "Late_blight"
            plant, _, disease = res["prediction"]["label"].partition("___")
            st.success(f"**{plant}** — {disease.replace('_', ' ')}")
            st.metric("Confiance", f"{res['prediction']['confidence']:.1%}")
            st.write("Top 3 :")
            st.table(res["top_k"])
        else:
            st.error(f"Erreur {r.status_code} : {r.text}")