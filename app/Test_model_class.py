import spacy
import spacy
import numpy as np
import joblib
from functions import transform_texte, SupervisedModel, UnsupervisedModel
import tensorflow_hub as hub

# Define corpus 
corpus = """Python is a high-level, general-purpose programming language.
 Its design philosophy emphasizes code readability with the use of significant indentation. 
 Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms,
  including structured, object-oriented and functional programming."""
# Modèle
nlp = spacy.load("en_core_web_sm")
# PartOfSpeech
pos_list = ["NOUN", "PROPN"]
# Cleaning corpus
text_cleaned = transform_texte(nlp, corpus, pos_list)
print("Texte cleane : ", text_cleaned)
# Supervised prediction
supervised_model = SupervisedModel()
supervised_pred = supervised_model.predict_tags(text_cleaned)
# Unsupervised prediction
unsupervised_model = UnsupervisedModel()
unsupervised_pred = unsupervised_model.predict_tags(text_cleaned)
# Result
print("Supervised : ", supervised_pred)
print("Unsupervised : ", unsupervised_pred)