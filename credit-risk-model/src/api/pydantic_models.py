from pydantic import BaseModel
from typing import Optional

class PredictionInput(BaseModel):
    Recency: float
    Frequency: float
    MonetaryTotal: float
    MonetaryMean: float
    MonetaryMax: float
    MonetaryMin: float
    MonetaryStd: float
    ProductCategory: str
    ChannelId: str

class PredictionOutput(BaseModel):
    risk_probability: float
    credit_score: int
    optimal_amount: float
    optimal_duration: int