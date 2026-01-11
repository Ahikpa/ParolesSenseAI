# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Importe l'instance unique de notre modèle de recommandation
# Ce fichier (main.py) et model.py sont dans le même dossier 'backend'
from .model import recommendation_model

app = FastAPI(
    title="API de Recommandation de Chansons",
    description="Une API pour obtenir des recommandations de chansons basées sur la similarité sémantique des paroles.",
    version="1.0.0"
)

# --- Middleware CORS ---
# Permet à notre frontend (Streamlit) de communiquer avec cette API,
# même s'ils tournent sur des ports différents.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (pour le développement)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)

@app.on_event("startup")
async def startup_event():
    """Message au démarrage du serveur."""
    print("API prête à recevoir des requêtes.")
    print("Endpoints disponibles : /songs, /recommendations/{song_title}")

@app.get("/songs", response_model=List[str])
async def get_songs():
    """
    Retourne la liste de tous les titres de chansons disponibles.
    """
    try:
        song_list = recommendation_model.get_song_list()
        return song_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/{song_title}", response_model=List[str])
async def get_recommendations_for_song(song_title: str):
    """
    Retourne une liste de recommandations pour un titre de chanson donné.
    """
    try:
        recommendations = recommendation_model.get_recommendations(song_title)
        if not recommendations and song_title not in recommendation_model.get_song_list():
             raise HTTPException(status_code=404, detail="Chanson non trouvée.")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Pour exécuter ce serveur ---
# 1. Assurez-vous d'être dans le dossier qui contient le dossier 'backend'
# 2. Installez les dépendances : pip install "fastapi[all]"
# 3. Lancez le serveur avec uvicorn : uvicorn backend.main:app --reload
