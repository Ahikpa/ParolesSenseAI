# -*- coding: utf-8 -*-
# recommendation_system.py

"""
Ce script implémente un système de recommandation de chansons basé sur le contenu.
Il compare deux approches de vectorisation de texte :
1.  Approche classique : TF-IDF
2.  Approche moderne : Embeddings de phrases avec SentenceTransformers

Le script suit les étapes suivantes :
1.  Chargement et nettoyage des données textuelles (paroles de chansons).
2.  Vectorisation des textes selon les deux approches.
3.  Calcul de la similarité cosinus pour chaque approche.
4.  Génération de recommandations pour une chanson donnée.
5.  Visualisation des embeddings en 2D avec PCA.
"""

# =============================================================================
# Section 0: Imports et Configuration
# =============================================================================
import os
import glob
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Configuration initiale de NLTK ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Téléchargement des stopwords NLTK...")
    nltk.download('stopwords')

stop_words_fr = set(stopwords.words('french'))

# =============================================================================
# Section 1: Chargement et Prétraitement des Données
# =============================================================================
def clean_text(text):
    """Nettoie une chaîne de caractères : minuscules, sans ponctuation/chiffres/stopwords."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words_fr]
    return ' '.join(words)

def load_and_clean_data(data_directory):
    """Charge les chansons depuis un dossier, les nettoie et les retourne dans un DataFrame."""
    print(f"Chargement des données depuis : {data_directory}")
    filepaths = glob.glob(os.path.join(data_directory, '*.txt'))
    
    if not filepaths:
        print(f"ERREUR: Aucun fichier .txt n'a été trouvé dans le dossier '{data_directory}'.")
        print("Veuillez vérifier que le chemin est correct et que les fichiers existent.")
        return pd.DataFrame()

    chansons_data = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lyrics = f.read()
            title = os.path.basename(filepath).replace('.txt', '').replace('_', ' ').title()
            chansons_data.append({'titre': title, 'paroles': lyrics})
        except Exception as e:
            print(f"Erreur de lecture du fichier {filepath}: {e}")

    df = pd.DataFrame(chansons_data)
    print(f"{len(df)} chansons chargées.")
    
    print("Nettoyage des paroles en cours...")
    df['paroles_cleaned'] = df['paroles'].apply(clean_text)
    
    return df

# =============================================================================
# Section 2: Vectorisation (Les deux approches)
# =============================================================================

# --- Approche 1: TF-IDF ---
def vectorize_tfidf(texts):
    """Vectorise une liste de textes en utilisant TF-IDF."""
    print("\n--- Approche 1: Vectorisation avec TF-IDF ---")
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    print(f"Matrice TF-IDF créée. Dimensions : {matrix.shape}")
    return matrix

# --- Approche 2: Embeddings (Moderne) ---
def vectorize_embeddings(texts):
    """
    Vectorise une liste de textes en utilisant un modèle de SentenceTransformer.
    Cette approche capture le sens sémantique des paroles.
    """
    print("\n--- Approche 2: Vectorisation avec Embeddings Sémantiques ---")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERREUR: La librairie 'sentence-transformers' n'est pas installée.")
        print("Veuillez l'installer avec la commande : pip install sentence-transformers")
        return None

    # On utilise un modèle pré-entraîné pour le français.
    # 'distiluse-base-multilingual-cased-v1' est un bon choix polyvalent.
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    print("Modèle d'embedding chargé. Création des embeddings en cours...")
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    print(f"Matrice d'embeddings créée. Dimensions : {embeddings.shape}")
    return embeddings

# =============================================================================
# Section 3: Logique de Recommandation
# =============================================================================
def get_recommendations(song_title, df, similarity_matrix, top_n=5):
    """Génère des recommandations pour un titre de chanson donné."""
    if song_title not in df['titre'].values:
        print(f"ERREUR: Le titre '{song_title}' n'est pas dans la liste des chansons.")
        return

    # Trouver l'index de la chanson
    idx = df.index[df['titre'] == song_title].tolist()[0]
    
    # Obtenir les scores de similarité pour cette chanson
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Trier les chansons par similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtenir les indices des N chansons les plus similaires (en excluant elle-même)
    top_song_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    # Retourner les titres
    recommendations = df['titre'].iloc[top_song_indices]
    
    print(f"\nRecommandations pour '{song_title}':")
    for i, rec_title in enumerate(recommendations):
        print(f"{i+1}. {rec_title}")

# =============================================================================
# Section 4: Visualisation
# =============================================================================
def visualize_embeddings_pca(embedding_matrix, df):
    """Réduit la dimension des embeddings à 2D avec PCA et crée un graphique."""
    print("\n--- Visualisation: Réduction de dimension avec PCA ---")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embedding_matrix)
    
    plt.figure(figsize=(14, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # Annoter quelques points pour la lisibilité
    for i, title in enumerate(df['titre']):
        if i % max(1, len(df)//10) == 0: # Annoter environ 10 chansons
             plt.annotate(title, (embeddings_2d[i, 0], embeddings_2d[i, 1]), alpha=0.8)
    
    plt.title("Visualisation des Chansons (Embeddings projetés en 2D via PCA)")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.grid(True)
    
    # Sauvegarder le graphique
    output_filename = "visualisation_chansons_pca.png"
    plt.savefig(output_filename)
    print(f"Le graphique de visualisation a été sauvegardé sous : {output_filename}")


# =============================================================================
# Section 5: Exécution Principale
# =============================================================================
if __name__ == "__main__":
    # Définir le chemin vers le dossier contenant les .txt des chansons
    # Le script s'attend à ce que le dossier 'TP2/chansons' soit au même niveau que lui.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chansons_directory = os.path.join(current_dir, 'TP2', 'chansons')

    # 1. Charger et nettoyer les données
    df_chansons = load_and_clean_data(chansons_directory)

    if not df_chansons.empty:
        # Choisir une chanson pour tester les recommandations
        test_song_title = df_chansons['titre'].iloc[0]

        # --- Exécution de l'Approche 1: TF-IDF ---
        tfidf_matrix = vectorize_tfidf(df_chansons['paroles_cleaned'])
        cosine_sim_tfidf = cosine_similarity(tfidf_matrix)
        print("\n*** RÉSULTATS AVEC TF-IDF ***")
        get_recommendations(test_song_title, df_chansons, cosine_sim_tfidf)

        # --- Exécution de l'Approche 2: Embeddings ---
        embedding_matrix = vectorize_embeddings(df_chansons['paroles_cleaned'])
        if embedding_matrix is not None:
            cosine_sim_embed = cosine_similarity(embedding_matrix)
            print("\n*** RÉSULTATS AVEC EMBEDDINGS (APPROCHE MODERNE) ***")
            get_recommendations(test_song_title, df_chansons, cosine_sim_embed)
            
            # --- Visualisation (uniquement pour les embeddings) ---
            visualize_embeddings_pca(embedding_matrix, df_chansons)

    print("\nScript terminé.")
