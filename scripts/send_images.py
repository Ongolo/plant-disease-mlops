"""Envoie toutes les images d'un dossier a l'API (simulation d'utilisateurs reels).

Usage : python scripts/send_images.py --url http://35.181.112.52:8000 --folder demo_images/plantdoc
        (le token est lu dans la variable d'environnement API_TOKEN)
"""
import argparse
import collections
import os
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--folder", required=True)
args = p.parse_args()

headers = {"Authorization": f"Bearer {os.environ['API_TOKEN']}"}
files = sorted(f for f in Path(args.folder).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
codes, confidences = collections.Counter(), []
for f in files:
    r = requests.post(f"{args.url}/predict", headers=headers,
                      files={"file": (f.name, f.read_bytes(), "image/jpeg")}, timeout=60)
    codes[r.status_code] += 1
    if r.ok:
        confidences.append(r.json()["prediction"]["confidence"])
print(f"{len(files)} images envoyees ; codes HTTP : {dict(codes)}")
if confidences:
    print(f"confiance moyenne : {sum(confidences) / len(confidences):.3f}")