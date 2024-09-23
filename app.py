import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import re
import nltk
from nltk.corpus import wordnet
import streamlit as st

# Assurez-vous d'avoir NLTK installé et d'avoir téléchargé WordNet
nltk.download('wordnet')

# Fonction pour récupérer le contenu principal
def get_main_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie que la requête a réussi
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('article') or soup.find('div', {'class': 'main-content'})
        return main_content.get_text(strip=True) if main_content else "Contenu principal non trouvé."
    except Exception as e:
        return f"Erreur lors de la récupération de {url}: {e}"

# Titre de l'application
st.title("Analyse de la pertinence des URLs avec BERT")

# Uploader le fichier
uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=["xls", "xlsx"])

# Saisie de la requête
query = st.text_input("Entrez votre requête de recherche :", "consultant seo")

if uploaded_file is not None:
    # Lire le fichier Excel
    df_urls = pd.read_excel(uploaded_file)
    st.write("Contenu du fichier Excel :")
    st.write(df_urls)  # Affichez le contenu du DataFrame

    # Afficher les noms des colonnes
    st.write("Noms des colonnes :", df_urls.columns.tolist())

    # Vérifiez que la colonne 'Address' existe
    if 'Address' in df_urls.columns:
        # Créer une liste pour stocker les résultats
        data = []

        # Récupérer le contenu pour chaque URL
        for url in df_urls['Address']:  # Changer ici aussi
            content = get_main_content(url)
            data.append({"URL": url, "Contenu": content})

        # Créer un DataFrame à partir des données
        df = pd.DataFrame(data)

        # Charger le modèle BERT pré-entraîné
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Encoder la requête
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Calculer les scores de pertinence pour chaque contenu
        scores = []
        keywords_to_add = []
        keywords_count = []

        # Fonction pour obtenir les synonymes
        def get_synonyms(word):
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())  # Ajoute les synonymes
            return synonyms

        # Fonction pour convertir un dictionnaire en chaîne de caractères
        def dict_to_str(counts):
            return ', '.join([f"{k}: {v}" for k, v in counts.items()])

        # Extraire les mots-clés de la requête
        query_keywords = query.lower().split()
        query_word_counts = Counter(query_keywords)

        for content in df['Contenu']:
            content_embedding = model.encode(content, convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, content_embedding).item()
            scores.append(score)

            # Nettoyer et analyser les mots du contenu
            content_cleaned = re.findall(r'\w+', content.lower())
            content_word_counts = Counter(content_cleaned)
            
            # Identifier les mots-clés manquants et calculer les occurrences à ajouter
            missing_keywords = {}
            for keyword, count in query_word_counts.items():
                current_count = content_word_counts.get(keyword, 0)
                if current_count < count:
                    missing_keywords[keyword] = count - current_count
                    
                    # Ajouter les synonymes
                    for synonym in get_synonyms(keyword):
                        if synonym not in content_word_counts:
                            missing_keywords[synonym] = 1  # Proposer d'ajouter un synonyme

            keywords_to_add.append(", ".join(missing_keywords.keys()))
            keywords_count.append(missing_keywords)

        # Ajouter les scores, mots-clés et occurrences au DataFrame
        df['Score de pertinence'] = scores
        df['Mots-clés à ajouter'] = keywords_to_add
        df['Occurrences à ajouter'] = [dict_to_str(counts) for counts in keywords_count]

        # Afficher le DataFrame avec les scores et suggestions de mots-clés
        st.write("Résultats de l'analyse :")
        st.write(df[['URL', 'Score de pertinence', 'Mots-clés à ajouter', 'Occurrences à ajouter']])

        # Optionnel : Sauvegarder le DataFrame dans un fichier CSV
        csv = df.to_csv(index=False)
        st.download_button("Télécharger les résultats", csv, "pertinence_scores_with_keywords.csv")
    else:
        st.error("La colonne 'Address' n'existe pas dans le fichier Excel.")
