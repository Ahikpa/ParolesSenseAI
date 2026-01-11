# backend/Dockerfile

# Étape 1: Utiliser une image Python officielle et légère comme base
FROM python:3.11-slim

# Étape 2: Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3: Copier le fichier des dépendances et les installer
# Copier d'abord ce fichier permet à Docker de mettre en cache cette couche
# si les dépendances ne changent pas, accélérant les builds futurs.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 4: Copier le code de l'application et les données nécessaires
# On copie le dossier backend qui contient main.py et model.py
COPY ./backend ./backend
# On copie aussi le dossier des chansons pour que le modèle puisse le lire
COPY ./TP2 ./TP2

# Étape 5: Exposer le port sur lequel l'API va tourner
EXPOSE 8000

# Étape 6: La commande pour lancer l'application
# On utilise --host 0.0.0.0 pour rendre le serveur accessible depuis l'extérieur du conteneur.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
