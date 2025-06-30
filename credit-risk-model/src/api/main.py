from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load the model from MLflow
model = mlflow.pyfunc.load_model("model_path")  # Replace "model_path" with the actual model path

class CustomerData(BaseModel):
    Feature1: float
    Feature2: int
    # Add other features based on your model

@app.post("/predict")
def predict_risk(customer: CustomerData):
    data = pd.DataFrame([customer.dict()])
    risk_probability = model.predict(data)
    return {"risk_probability": risk_probability.tolist()}