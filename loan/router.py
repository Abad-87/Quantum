"""
loan/router.py
FastAPI router for Banking & Loan Approval.
Endpoint: POST /loan/predict
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
import logging

from .model_loader import get_model
from .predictor import predict
from fairness.checker import run_fairness_check
from utils.logger import log_prediction
from utils.database import save_prediction

router = APIRouter()
logger = logging.getLogger("loan.router")


# ─── Input Schema ─────────────────────────────────────────────────────────────
class LoanRequest(BaseModel):
    # ── Prediction features ───────────────────────────────────────────────────
    credit_score: int        = Field(..., ge=300,  le=850,    description="FICO credit score")
    annual_income: float     = Field(..., ge=0,               description="Annual income in USD")
    loan_amount: float       = Field(..., ge=100,             description="Requested loan amount in USD")
    loan_term_months: int    = Field(..., ge=6,    le=360,    description="Repayment period in months")
    employment_years: float  = Field(..., ge=0,    le=50,     description="Years at current employer")
    existing_debt: float     = Field(0.0, ge=0,              description="Current outstanding debt in USD")
    num_credit_lines: int    = Field(0,   ge=0,   le=50,     description="Number of open credit accounts")

    # ── Sensitive attributes (fairness monitoring only) ───────────────────────
    gender: str | None       = Field(None, description="Fairness monitoring only")
    religion: str | None     = Field(None, description="Fairness monitoring only")
    ethnicity: str | None    = Field(None, description="Fairness monitoring only")
    age_group: str | None    = Field(None, description="Fairness monitoring only (e.g. '18-25', '26-40')")

    @field_validator("loan_amount")
    @classmethod
    def loan_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("loan_amount must be greater than 0")
        return v


# ─── Output Schema ────────────────────────────────────────────────────────────
class LoanResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    explanation: str
    fairness: dict
    message: str


# ─── Endpoint ─────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=LoanResponse)
async def loan_predict(request: LoanRequest, background_tasks: BackgroundTasks):
    """
    **Loan Approval Prediction**

    Evaluates a loan application based on financial features only.
    Sensitive attributes (gender, religion, ethnicity, age_group) are used
    ONLY for fairness auditing and never influence the model decision.

    **Returns:** approval decision, confidence, explanation, fairness report.
    """

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Extract prediction features ───────────────────────────────────────────
    prediction_features = {
        "credit_score":      request.credit_score,
        "annual_income":     request.annual_income,
        "loan_amount":       request.loan_amount,
        "loan_term_months":  request.loan_term_months,
        "employment_years":  request.employment_years,
        "existing_debt":     request.existing_debt,
        "num_credit_lines":  request.num_credit_lines,
    }

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        prediction, confidence, explanation = predict(model, prediction_features)
    except Exception as e:
        logger.error(f"Loan prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    prediction_label = "Approved" if prediction == 1 else "Rejected"

    # ── Fairness check ────────────────────────────────────────────────────────
    sensitive_attr  = None
    sensitive_value = None

    for attr, val in [
        ("gender", request.gender),
        ("religion", request.religion),
        ("ethnicity", request.ethnicity),
        ("age_group", request.age_group),
    ]:
        if val:
            sensitive_attr, sensitive_value = attr, val
            break

    fairness_result = run_fairness_check(
        prediction=prediction,
        sensitive_attr=sensitive_attr or "not_provided",
        sensitive_value=sensitive_value or "unknown",
        domain="loan",
    )

    # Strip sensitive_value from public response
    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    # ── Log & persist (background) ────────────────────────────────────────────
    log_record = {
        "domain": "loan",
        "input": prediction_features,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "explanation": explanation,
        "fairness": safe_fairness,
    }
    background_tasks.add_task(log_prediction, **{
        "domain": "loan",
        "input_data": prediction_features,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "explanation": explanation,
        "fairness_result": fairness_result,
    })
    background_tasks.add_task(save_prediction, log_record)

    # ── Response ──────────────────────────────────────────────────────────────
    warning_msg = ""
    if fairness_result.get("warning"):
        warning_msg = f" ⚠️ Fairness Warning: {fairness_result['warning']}"

    return LoanResponse(
        prediction=prediction,
        prediction_label=prediction_label,
        confidence=confidence,
        explanation=explanation,
        fairness=safe_fairness,
        message=f"Prediction complete.{warning_msg}",
    )
