# backend/model.py
import os
import glob
import re
import string
import pandas as pd
from nltk.corpus import stopwords
import nltk

# --- Configuration de NLTK ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
finally:
    stop_words_fr = set(stopwords.words('french'))

def clean_text(text):
    """Nettoie une chaîne de caractères."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words_fr]
    return ' '.join(words)

class RecommendationModel:
    """
    Classe encapsulant le modèle de recommandation.
    Charge les données et le modèle une seule fois.
    """
    def __init__(self):
        self.df = None
        self.model = None
        self.similarity_matrix = None
        self._load_data()
        self._load_model_and_build_matrix()

    def _load_data(self):
        """Charge et nettoie les données des chansons."""
        print("Chargement des données...")
        # Chemin relatif depuis le script model.py -> ../TP2/chansons
        current_dir = os.path.dirname(__file__)
        chansons_dir = os.path.join(current_dir, '..', 'TP2', 'chansons')
        
        filepaths = glob.glob(os.path.join(chansons_dir, '*.txt'))
        if not filepaths:
            raise FileNotFoundError(f"Aucun fichier .txt trouvé dans {chansons_dir}")

        chansons_data = []
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                lyrics = f.read()
            title = os.path.basename(filepath).replace('.txt', '').replace('_', ' ').title()
            chansons_data.append({'titre': title, 'paroles': lyrics})
        
        self.df = pd.DataFrame(chansons_data)
        self.df['paroles_cleaned'] = self.df['paroles'].apply(clean_text)
        print(f"{len(self.df)} chansons chargées.")

    def _load_model_and_build_matrix(self):
        """Charge le modèle d'embedding et pré-calcule la matrice de similarité."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("Veuillez installer les librairies requises : pip install sentence-transformers scikit-learn")

        print("Chargement du modèle d'embedding (cela peut prendre un moment)...")
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        
        print("Création de la matrice de similarité...")
        embeddings = self.model.encode(self.df['paroles_cleaned'].tolist(), show_progress_bar=True)
        self.similarity_matrix = cosine_similarity(embeddings)
        print("Matrice de similarité prête.")

    def get_song_list(self):
        """Retourne la liste des titres de chansons."""
        return self.df['titre'].tolist()

    def get_recommendations(self, song_title, top_n=5):
        """Retourne des recommandations pour un titre de chanson donné."""
        if song_title not in self.df['titre'].values:
            return []

        idx = self.df.index[self.df['titre'] == song_title].tolist()[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_song_indices = [i[0] for i in sim_scores[1:top_n+1]]
        
        recommendations = self.df['titre'].iloc[top_song_indices].tolist()
        return recommendations

# Création d'une instance unique qui sera importée par l'API
recommendation_model = RecommendationModel()
