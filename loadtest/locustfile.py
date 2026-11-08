import os

from locust import HttpUser, between, task

# Token de l'utilisateur dedie au test de charge (bob) et image envoyee
TOKEN = os.environ["API_TOKEN"]
IMG_PATH = os.environ.get("IMG_PATH", "loadtest/feuille_test.jpg")

with open(IMG_PATH, "rb") as f:
    IMG_BYTES = f.read()


class PlantApiUser(HttpUser):
    """Un utilisateur virtuel : envoie une photo, attend 1 a 2 s, recommence."""
    wait_time = between(1, 2)

    @task(10)
    def predict(self):
        self.client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("feuille.jpg", IMG_BYTES, "image/jpeg")},
            name="/predict",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
