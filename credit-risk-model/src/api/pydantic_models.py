from pydantic import BaseModel

class CustomerData(BaseModel):
    Feature1: float
    Feature2: int
    # Add other features based on your model