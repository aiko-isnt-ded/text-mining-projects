from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

api_model = pipeline('text-classification', model='aiko-isnt-ded/distilbert-demo-model') # Nombre de usuario y del repo

class TextInput(BaseModel):
    text: str # Para asegurarse de que lea lo que le enviamos como texto

@app.post('/predict')
def predict(data: TextInput):
    resultado = api_model(data.text)
    return {'result': resultado}