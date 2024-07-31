from fastapi import FastAPI
from fastapi.responses import JSONResponse
from rag import rag_func
from pydantic import BaseModel
import pickle
import numpy as np
from diabetes import diabetes_convo

filename = 'rf_model.sav'
with open(filename, 'rb') as f:
   model = pickle.load(f)
with open('normalizer.pkl', 'rb') as f:
     normalizer = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
     scaler = pickle.load(f)
app = FastAPI()
class Prompt(BaseModel):
    content: str
class Data(BaseModel):
    pregnancies: float
    glucose: float
    bp: float
    sknthck: float
    insulin: float
    bmi: float
    pdf: float
    age: float

@app.post("/rag")
async def rag(prompt: Prompt):
    response_data = rag_func(prompt)
    return response_data

@app.post("/classification")
async def classification(data: Data):

    x_val = np.array([[data.pregnancies, data.glucose, data.bp, data.sknthck, data.insulin, data.bmi, data.pdf, data.age]])

    x_normalized = normalizer.transform(x_val)

    x_scaled = scaler.transform(x_normalized)

    with open('rf_model.sav', 'rb') as f:
        rf = pickle.load(f)

    probabilities = rf.predict_proba(x_scaled)[0]

    predicted_class = np.argmax(probabilities)
    predicted_probability = probabilities[predicted_class]
    
    
    return diabetes_convo(data,int(predicted_class),predicted_probability*100)
