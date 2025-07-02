# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import mlflow
import pandas as pd
import numpy as np
import os
import json

from src.data_processing import DateFeatureExtractor, AggregateFeatureCreator # Ensure these are imported
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="Credit Risk Model API",
    description="API for predicting credit risk of customers."
)
MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production" # or "Staging", or "1" for a specific version

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model '{MODEL_NAME}' ({MODEL_STAGE}) loaded successfully from MLflow.")
except Exception as e:
    print(f"Error loading model from MLflow: {e}")
    print("Attempting to load a dummy model for development/testing purposes.")
    class DummyModel:
        def predict_proba(self, X):
            # Simulate probabilities (e.g., lower for low risk, higher for high risk)
            # Assuming X has a 'TransactionAmount' and 'CustomerIncome' column for dummy logic
            if isinstance(X, pd.DataFrame):
                # Simple dummy logic: higher risk for low income and low transaction amount
                # This is just for demonstration, actual model logic is more complex.
                return np.array([[0.9, 0.1] if row['CustomerIncome'] < 40000 and row['TransactionAmount'] < 100 else [0.2, 0.8] for _, row in X.iterrows()])
            return np.array([[0.5, 0.5]]) # Default if input format unknown

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    model = DummyModel()
    print("Loaded a dummy model. Please ensure MLflow model is correctly registered and accessible in production.")


@app.get("/")
async def root():
    return {"message": "Credit Risk Model API is running. Go to /docs for API documentation."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(features: CustomerFeatures):

    try:
        input_data = pd.DataFrame([features.model_dump()])

        input_df = pd.DataFrame([features.model_dump()])
        # Drop CustomerId as it's not a feature for prediction but useful for response
        customer_id_for_response = input_df['CustomerId'].iloc[0]
        input_df = input_df.drop(columns=['CustomerId'])

        high_risk_probability = model.predict_proba(input_df)[:, 1][0]
        prediction = int(model.predict(input_df)[0])

        return PredictionResponse(
            customer_id=customer_id_for_response,
            high_risk_probability=high_risk_probability,
            prediction=prediction
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
