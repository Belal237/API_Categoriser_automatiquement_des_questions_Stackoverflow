#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: belal_oumar
"""

# Contient l'ensemble des fonctions nécessaires
# pour l'exécution du main_code

# Parsing des données texte
from bs4 import BeautifulSoup

# Librairie nltk pour traiter les mots
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Data Persistence
import joblib
# import tensorflow_hub as hub


def remove_html_syntax(x):
    """
    Function that remove HTML syntax in the document.
    And get the text without html syntax

    Parameters
    ----------------------------------------
    x : string
        Sequence of characters to modify.
    ----------------------------------------
    """
    sp = BeautifulSoup(x, "lxml")
    no_html = sp.get_text()
    return no_html


def remove_pos(nlp, x, pos_list):
    """
    This function will remove the unecessary part of speech
    and keep only necessary part of speech given by pos_list
    nlp :
    Parameters
    ----------------------------------------
    nlp : model
    x : corpus, string, Sequence of characters to modify
    pos_list : part of speech we want to keep in the corpus.
    ----------------------------------------
    """
    doc = nlp(x)
    text_list = []
    for token in doc:
        if token.pos_ in pos_list:
            text_list.append(token.text)
    join_text = " ".join(text_list)
    join_text = join_text.lower().replace("c #", "c#")

    return join_text


def text_cleaner(x, nlp, pos_list):
    """
    Cleaning the corpus by removing punctuation, URLS,
    english contractions numbers

    """
    # Remove unecessary POS not includ un pos_list
    x = remove_pos(nlp, x, pos_list)
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove URLs
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra whitespaces
    x = re.sub('\s+', ' ', x)
    # Return cleaned text
    return x


def tokenizer_fct(sentence):
    # Tokenize the corpus using Word_tokenize
    word_tokens = word_tokenize(sentence)
    return word_tokens


# Stop words
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']


def stop_word_filter_fct(list_words):
    filtered_w = [w for w in list_words if w not in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2


# Lemmatizer (base d'un mot)
def lemma_fct(list_words):
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w


def transform_texte(nlp, corpus, pos_list):
    """
    This fucntion allow us to make all the transformation in the texte
    by lemmatized them, remove punctuations, url & html syntax...
    :param nlp: Method used : spacy - en_core_web_sm
    :param corpus: The corpus to transform
    :param pos_list: Part Of Speech we want to keep
    :return: corpus cleaned
    """
    new_corpus = remove_pos(nlp, corpus, pos_list)
    corpus_clean = text_cleaner(new_corpus, nlp, pos_list)
    word_tokens = tokenizer_fct(corpus_clean)
    sw = stop_word_filter_fct(word_tokens)
    lem_w = lemma_fct(sw)
    test_cleaned = ' '.join(lem_w)
    return test_cleaned

# def vectorisation_USE(sentences):
#     batch_size = 10
#     embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#     for step in range(len(sentences) // batch_size + 1):
#         idx = step * batch_size
#         feat = embed(sentences[idx: idx + batch_size])

#         if step == 0:
#             feature = feat
#         else:
#             feature = np.concatenate((feature, feat))
#     return feature


class SupervisedModel:

    # Definition et appel des models deja entrainé
    def __init__(self):
        filename_supervised_model_tfidf = "./Models/model_tfidf_lr.joblib"
        #filename_supervised_model_use = "./Models/model_use_lr.joblib"
        filename_mlb_model = "./Models/multilabel.joblib"
        filename_tfidf_vectorizer = "./Models/tfidf_vectorizer.joblib"

        self.supervised_model_tfidf = joblib.load(open(filename_supervised_model_tfidf, 'rb'))
        #self.supervised_model_use = joblib.load(open(filename_supervised_model_use, 'rb'))
        self.mlb_model = joblib.load(open(filename_mlb_model, 'rb'))
        self.tfidf_vectorizer = joblib.load(open(filename_tfidf_vectorizer, 'rb'))

    
    # Fonction de prediction de tags
    def predict_tags(self, text):
        """
        Prediction des tags à partir du texte lemmatizer.
        
        Args:
            supervised_model(): Model pour prediction
            mlb_model(): Multilabel pour obtenir les tags
        Returns:
            tags(list): Liste des tags predits par le model
        """
        text_list = [text]
        vectorisation = self.tfidf_vectorizer.transform(text_list)
        #vectorisation = vectorisation_USE(text_list)
        input_vector = pd.DataFrame(vectorisation.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        #input_vector = vectorisation.numpy()
        #res = self.supervised_model_use.predict(input_vector)
        res = self.supervised_model_tfidf.predict(input_vector)
        res = self.mlb_model.inverse_transform(res)
        tag = list({tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
        return tag
        
class UnsupervisedModel:

    def __init__(self):
        filename_model = "./Models/LDA_model.joblib"
        filename_dictionary = "./Variables/common_dictionary.joblib"

        self.unsupervisedmodel = joblib.load(open(filename_model, 'rb'))
        self.dictionary = joblib.load(open(filename_dictionary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags of a preprocessed text
        
        Args:
            text(list): preprocessed text
        Returns:
            res(list): list of tags
        """
        split_corp = text.split()
        new_corpus = self.dictionary.doc2bow(split_corp)
        topics = self.unsupervisedmodel.get_document_topics(new_corpus)
        
        #find most relevant topic according to probability
        relevant_topic = topics[0][0]
        relevant_topic_prob = topics[0][1]
        
        for i in range(len(topics)):
            if topics[i][1] > relevant_topic_prob:
                relevant_topic = topics[i][0]
                relevant_topic_prob = topics[i][1]

                
        #retrieve associated to topic tags present in submited text
        res = self.unsupervisedmodel.get_topic_terms(topicid=relevant_topic)
        tag = [self.dictionary[tag[0]] for tag in res if self.dictionary[tag[0]] in text]

        return tag
