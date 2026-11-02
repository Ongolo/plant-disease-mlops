# Image de base : Linux minimal avec Python 3.11
FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# On installe les dépendances AVANT de copier le code (cache Docker plus efficace)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie le code de l'API
COPY app ./app

# Le port sur lequel l'API écoute
EXPOSE 8000

# Commande lancée au démarrage du conteneur
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]