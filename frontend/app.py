# frontend/app.py
import streamlit as st
import requests
import urllib.parse
import os

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Système de Recommandation de Chansons",
    page_icon="🎵",
    layout="centered"
)

# L'URL de l'API est maintenant configurable via une variable d'environnement.
# Par défaut, elle pointe vers l'URL locale, mais en production (Docker), 
# nous la définirons sur l'adresse du service backend (ex: http://backend:8000).
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- Fonctions pour communiquer avec l'API ---

def get_all_songs():
    """Récupère la liste de toutes les chansons depuis l'API."""
    try:
        response = requests.get(f"{API_URL}/songs")
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        st.warning("Veuillez vous assurer que le serveur backend est bien lancé. (Voir les instructions dans le README)")
        return None

def get_recommendations(song_title):
    """Demande les recommandations pour une chanson donnée à l'API."""
    # Encoder le titre pour qu'il soit sûr à passer dans une URL
    encoded_title = urllib.parse.quote(song_title)
    try:
        with st.spinner(f"Recherche de recommandations pour '{song_title}'..."):
            response = requests.get(f"{API_URL}/recommendations/{encoded_title}")
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des recommandations : {e}")
        return None

# --- Interface Utilisateur ---

st.title("🎵 Système de Recommandation de Chansons")
st.write(
    "Choisissez une chanson dans la liste ci-dessous pour découvrir des titres similaires "
    "basés sur l'analyse sémantique de leurs paroles."
)

# Récupérer la liste des chansons pour le menu déroulant
song_list = get_all_songs()

if song_list:
    # Créer le menu déroulant et le bouton
    selected_song = st.selectbox(
        "Choisissez une chanson :",
        options=song_list
    )

    if st.button("Obtenir les recommandations"):
        if selected_song:
            recommendations = get_recommendations(selected_song)
            
            if recommendations:
                st.success(f"Voici les 5 chansons recommandées similaires à **{selected_song}** :")
                # Affichage sous forme de liste numérotée
                for i, rec in enumerate(recommendations):
                    st.markdown(f"**{i+1}.** {rec}")
            else:
                # Gérer le cas où la liste de recommandations est vide mais sans erreur
                st.info("Aucune recommandation trouvée pour cette chanson.")

# --- Instructions pour l'utilisateur ---
st.markdown("---")
with st.expander("Comment lancer cette application ?"):
    st.markdown("""
    Cette application est composée de deux parties : un **backend** (le moteur) et un **frontend** (cette interface).
    
    1.  **Lancer le Backend (le moteur d'API) :**
        *   Ouvrez un premier terminal.
        *   Assurez-vous d'être dans le dossier racine du projet.
        *   Lancez la commande : `uvicorn backend.main:app --reload`
        *   Attendez de voir le message indiquant que le serveur est prêt.

    2.  **Lancer le Frontend (cette page) :**
        *   Ouvrez un **second** terminal.
        *   Assurez-vous d'être dans le dossier racine du projet.
        *   Lancez la commande : `streamlit run frontend/app.py`
        
    Votre navigateur devrait s'ouvrir automatiquement sur cette page web.
    """)
