from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.components.model_registry import ModelRegistry
from src.config import CATEGORICAL_COLS, NUMERICAL_COLS

app = FastAPI()


registry = ModelRegistry()
model, preprocessor = registry.load_model_and_preprocessor()


encoders = preprocessor.encoders
scaler = preprocessor.scaler

class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def root():
    return {"message": "ANN Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    
    df = pd.DataFrame([data.dict()])

    
    for col in CATEGORICAL_COLS:
        df[col] = encoders[col].transform(df[col])

    
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

    
    prob = model.predict(df.values)[0][0]   
    prediction = int(prob > 0.5)

    return {
        "probability": float(prob),
        "prediction": prediction
    }