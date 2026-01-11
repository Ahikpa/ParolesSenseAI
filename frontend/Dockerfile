# frontend/Dockerfile

# Étape 1: Utiliser une image Python officielle et légère comme base
FROM python:3.11-slim

# Étape 2: Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3: Copier le fichier des dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 4: Copier le code de l'application frontend
COPY ./frontend ./frontend

# Étape 5: Exposer le port sur lequel Streamlit va tourner
EXPOSE 8501

# Étape 6: La commande pour lancer l'application Streamlit
# On utilise --server.address 0.0.0.0 pour le rendre accessible.
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
