import spacy
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from .functions import transform_texte, SupervisedModel, UnsupervisedModel

app = FastAPI(
    title='Application de prediction de Tags',
    description='Application nous permettant de faire des predictions pour des question via FastAPI + uvicorn',
    version='0.0.1')


@app.get("/")
def root():
    return {"Welcome to the API. Check /docs for usage"}


class Input(BaseModel):
    question: str



@app.post("/predict")
async def get_prediction(data: Input):
    # Creation du corpus
    corpus = data.question
    # Modèle
    nlp = spacy.load("en_core_web_sm")
    # PartOfSpeech
    pos_list = ["NOUN", "PROPN"]
    # Cleaning text
    text_cleaned = transform_texte(nlp, corpus, pos_list)
    # Supervised prediction
    supervised_model = SupervisedModel()
    supervised_pred = supervised_model.predict_tags(text_cleaned)
    # Unsupervised prediction
    unsupervised_model = UnsupervisedModel()
    unsupervised_pred = unsupervised_model.predict_tags(text_cleaned)
    # Encodage json
    question = jsonable_encoder(data.question)

    return JSONResponse(status_code=200,
                        content={"Question ": question,
                                 "Tags with Supervised Model ": supervised_pred,
                                 "Tags with Unsupervised Model ": unsupervised_pred})

