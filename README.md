# ParolesSenseAI : Système de Recommandation de Chansons par IA

**ParolesSenseAI** est une application web qui recommande des chansons en analysant le sens profond de leurs paroles. Basé sur des modèles de traitement du langage naturel (NLP) modernes, ce projet démontre comment l'IA peut comprendre la similarité thématique et émotionnelle entre des textes musicaux.

L'application permet de choisir une chanson dans un corpus et d'obtenir instantanément une liste de titres similaires, non pas sur la base de mots-clés communs, mais sur la base de leur "vibration" sémantique.

## ✨ Fonctionnalités

*   **Recommandation Sémantique :** Utilise des embeddings de phrases (`Sentence Transformers`) pour capturer le sens et le contexte des paroles.
*   **Comparaison de Modèles :** Le code initial (`recommendation_system.py`) compare l'approche moderne avec la méthode classique TF-IDF, prouvant la supériorité de l'analyse sémantique.
*   **Visualisation des Données :** Génère un graphique PCA pour visualiser la proximité thématique des chansons dans un espace 2D.
*   **Interface Utilisateur Interactive :** Une application Streamlit simple et intuitive pour interagir avec le modèle.
*   **Architecture Moderne :** Construit sur une API backend performante (FastAPI) découplée du frontend.
*   **Prêt pour le Déploiement :** Entièrement conteneurisé avec Docker et Docker Compose, prêt à être déployé sur des plateformes comme Dokploy.

## 🏗️ Architecture de l'Application

L'application suit une architecture découplée, standard dans le développement web moderne :

```
Utilisateur via Navigateur
       |
       v
+------------------------+
| Frontend (Streamlit)   |  (Tourne sur le port 8501)
| Interface Utilisateur  |
+------------------------+
       |
       v (Requête HTTP API)
+------------------------+
| Backend (FastAPI)      |  (Tourne sur le port 8000)
| - API REST             |
| - Modèle NLP chargé    |
+------------------------+
```

## 🛠️ Technologies Utilisées

*   **Backend :** FastAPI, Uvicorn, Sentence-Transformers, Scikit-learn, NLTK, Pandas
*   **Frontend :** Streamlit, Requests
*   **Déploiement :** Docker, Docker Compose

## 🚀 Lancer le Projet

Il y a deux manières de lancer le projet en local : avec Docker (recommandé) ou manuellement.

### Méthode 1 : Avec Docker (Recommandé)

C'est la méthode la plus simple et la plus fiable, car elle gère tout pour vous.

**Prérequis :** Avoir [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et en cours d'exécution.

1.  Clonez ce dépôt :
    ```sh
    git clone <URL_DE_VOTRE_REPO>
    cd <NOM_DU_REPO>
    ```

2.  Lancez l'application avec Docker Compose :
    ```sh
    docker-compose up --build
    ```
    Cette commande va construire les images du backend et du frontend, puis démarrer les deux services. Le premier build peut prendre quelques minutes pour télécharger les modèles.

3.  Accédez à l'application :
    Ouvrez votre navigateur et allez sur **`http://localhost:8501`**.

### Méthode 2 : Manuellement

Cette méthode vous permet de lancer les services séparément sans Docker.

1.  Clonez le dépôt et installez les dépendances :
    ```sh
    git clone <URL_DE_VOTRE_REPO>
    cd <NOM_DU_REPO>
    pip install -r requirements.txt
    ```

2.  **Lancez le Backend :**
    Ouvrez un premier terminal et exécutez :
    ```sh
    uvicorn backend.main:app --reload
    ```
    Laissez ce terminal ouvert.

3.  **Lancez le Frontend :**
    Ouvrez un *second* terminal et exécutez :
    ```sh
    streamlit run frontend/app.py
    ```
    Votre navigateur devrait s'ouvrir automatiquement sur la page de l'application.

## ☁️ Déploiement sur Dokploy

Ce projet est prêt à être déployé sur une plateforme comme Dokploy :

1.  Poussez votre code sur un dépôt Git (GitHub, GitLab...).
2.  Dans Dokploy, créez une nouvelle application et connectez-la à votre dépôt Git.
3.  Choisissez `docker-compose.yml` comme méthode de build/déploiement.
4.  Dokploy s'occupera de construire et de lancer vos services. Configurez le port `8501` (celui du frontend) comme port principal à exposer.

---
*Ce projet a été développé dans le cadre d'une formation en Data Science.*
