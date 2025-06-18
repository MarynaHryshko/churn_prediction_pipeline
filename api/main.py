from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("model/churn_model.pkl")

# Define the request body structure
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerFeatures):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Align with training format
    df_transformed = pd.get_dummies(df)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df_transformed:
            df_transformed[col] = 0  # Add missing dummy columns

    df_transformed = df_transformed[model_features]  # Order columns

    # Predict
    prediction = model.predict(df_transformed)[0]
    return {"churn_prediction": int(prediction)}
