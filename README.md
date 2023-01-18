# Introduction
Ce projet concerne la mise en place d'une place d'une API de prédiction/suggestion automatique des tags.

Il s'agit de développer une API de suggestion de tags à destination des utilisateur de [Stack Overflow](https://stackoverflow.com/). Le besoin étant de faciliter les débutants à leur suggérer des tags pertinents pour leur questions ou intérrogations.


Un [second repository](https://github.com/Belal237/Categoriser_automatiquement_des_questions_Stackoverflow) contient les travaux de pré-traitement des documents, l'analyse et l'entrainement des modèles.

Il était demandé de réaliser:

- Le fitrage des données issue de l'API [stackexchange explorer](https://data.stackexchange.com/stackoverflow/query/new)
- Réaliser le pétraitement des documents 
- Comparer des approches suppervisées (Regression logistique, Random Forest) et non supervisées (LDA, NMF) afin de prédire des tags
- Réaliser les fonctions et classes nécessaire à l'implémentation de l'API. 
- Développer une API et la mettre en production

# Contenu du repositiry:
- Models : Ce dossier contient l'ensembles modèles entrainé dans le [repository](https://github.com/Belal237/Categoriser_automatiquement_des_questions_Stackoverflow), il contient les modèles supervisés (OneVsRestClassifier avec différents estimateur : Régression Logistique et le RandomForest) entrainés sur des vecteurs obtenu à partir de la base de donnée des texte transformé via (TF-IDF, ou Word2Vec ou BERT ou USE) ainsi que les modèles non supervisés.
- Variables : contient les variables et le dictionnaires des mots pour l'application des modèles
- App : Ensemble des fonctions qui permettent de pouvoir executer l'API, ainsi que l'API.

# Stack technique:
- Python
- Spacy
- FastAPI
- Uvicorn
- Heroku

# Liens vers l'API:
- [Documentation OpenApi de l'API]()
- [Endpoint permettant de réaliser la rédiction de tags]()
