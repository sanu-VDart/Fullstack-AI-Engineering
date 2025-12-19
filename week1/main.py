from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Classifier API")

# Load model once at startup
model = joblib.load("model/model.pkl")

# Request schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Response schema
class PredictionOutput(BaseModel):
    predicted_class: int
    predicted_label: str

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    features = np.array([[ 
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_class": int(prediction),
        "predicted_label": TARGET_NAMES[prediction]
    }

    
