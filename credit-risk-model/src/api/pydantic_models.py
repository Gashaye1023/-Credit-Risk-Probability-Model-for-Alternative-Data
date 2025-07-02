# src/api/pydantic_models.py

from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd 

class CustomerFeatures(BaseModel):
    # Example features (adjust based on your final model features)
    TransactionDate: str = Field(..., example="2023-01-01") # Raw date string as input
    TransactionAmount: float = Field(..., example=150.75)
    MerchantCategory: str = Field(..., example="Retail")
    PaymentType: str = Field(..., example="Card")
    CustomerAge: Optional[int] = Field(None, example=35) # Optional if can be missing
    CustomerIncome: Optional[float] = Field(None, example=65000.0) # Optional if can be missing
    CreditScore_External: Optional[int] = Field(None, example=720) # Optional if can be missing
    CustomerId: int = Field(..., example=12345) # Need customer ID for RFM lookup if RFM is dynamic


class PredictionResponse(BaseModel):
    customer_id: int = Field(..., example=12345)
    high_risk_probability: float = Field(..., example=0.15)
    prediction: int = Field(..., example=0) # 0 for low risk, 1 for high risk