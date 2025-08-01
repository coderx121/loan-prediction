from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Loan Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local testing; restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow necessary methods
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
try:
    model = joblib.load("best_loan_model.pkl")
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")

# Input schema
class LoanInput(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Credit_History: float

# Serve frontend
@app.get("/")
async def serve_frontend():
    from fastapi.responses import FileResponse
    return FileResponse("static/loan_prediction.html")

# Prediction endpoint
@app.post("/predict")
async def predict_loan_status(data: LoanInput):
    # Validate input
    if data.ApplicantIncome < 0 or data.CoapplicantIncome < 0 or data.LoanAmount <= 0:
        raise HTTPException(status_code=400, detail="Invalid input: Income and LoanAmount must be non-negative, LoanAmount must be positive")
    if data.Credit_History not in [0, 1]:
        raise HTTPException(status_code=400, detail="Credit_History must be 0 or 1")

    # Derived features
    total_income = data.ApplicantIncome + data.CoapplicantIncome
    loan_income_ratio = data.LoanAmount / total_income if total_income > 0 else 0
    log_applicant_income = np.log1p(data.ApplicantIncome)
    log_loan_amount = np.log1p(data.LoanAmount)

    features = {
        "Log_ApplicantIncome": log_applicant_income,
        "Log_LoanAmount": log_loan_amount,
        "Credit_History": data.Credit_History,
        "Total_Income": total_income,
        "Loan_Income_Ratio": loan_income_ratio
    }

    input_df = pd.DataFrame([features])
    try:
        prediction = model.predict(input_df)[0]
        loan_status = "Approved" if prediction == 1 else "Rejected"
        return {"Loan_Status": loan_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")