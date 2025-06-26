from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import pandas as pd
import logging
from .pydantic_models import PredictionInput, PredictionOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk Prediction API")

# Load model
try:
    model = mlflow.sklearn.load_model("../models/best_model")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        risk_probability = float(model.predict_proba(input_df)[0, 1])
        
        # Convert probability to credit score (300-850 scale)
        credit_score = int(300 + (risk_probability * 550))
        
        return {
            "risk_probability": risk_probability,
            "credit_score": credit_score,
            "optimal_amount": input_data.MonetaryMean * 2,  # Simplified calculation
            "optimal_duration": min(30, int(input_data.Frequency * 7))  # Simplified calculation
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise