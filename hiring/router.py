"""
hiring/router.py

FastAPI router for the Job Hiring domain.

Endpoint: POST /hiring/predict

This is the main API surface. It:
1. Validates the incoming request.
2. Separates sensitive attributes from prediction features.
3. Calls the model for prediction.
4. Runs a fairness check (sensitive attrs used ONLY here).
5. Returns prediction + explanation + fairness report.
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
logger = logging.getLogger("hiring.router")


# ─── Input Schema ─────────────────────────────────────────────────────────────
class HiringRequest(BaseModel):
    """
    Input schema for hiring prediction.
    Sensitive attributes are clearly marked and will NOT be passed to the model.
    """

    # ── Prediction features (what the model sees) ────────────────────────────
    years_experience: float = Field(..., ge=0, le=50, description="Years of work experience")
    education_level: int    = Field(..., ge=0, le=3,  description="0=HS, 1=Bachelor, 2=Master, 3=PhD")
    technical_score: float  = Field(..., ge=0, le=100, description="Technical assessment score")
    communication_score: float = Field(..., ge=0, le=100, description="Communication assessment score")
    num_past_jobs: int      = Field(..., ge=0, le=30, description="Number of previous jobs")
    certifications: int     = Field(0,    ge=0, le=20, description="Number of relevant certifications")

    # ── Sensitive attributes (fairness evaluation ONLY) ──────────────────────
    gender: str | None   = Field(None, description="For fairness monitoring only — not used in prediction")
    religion: str | None = Field(None, description="For fairness monitoring only — not used in prediction")
    ethnicity: str | None = Field(None, description="For fairness monitoring only — not used in prediction")

    @field_validator("education_level")
    @classmethod
    def validate_education(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("education_level must be 0 (HS), 1 (Bachelor), 2 (Master), or 3 (PhD)")
        return v


# ─── Output Schema ────────────────────────────────────────────────────────────
class HiringResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    explanation: str
    fairness: dict
    message: str


# ─── Endpoint ─────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=HiringResponse)
async def hiring_predict(request: HiringRequest, background_tasks: BackgroundTasks):
    """
    **Job Hiring Prediction**

    Predicts whether a candidate should be hired based on objective features.
    Sensitive attributes (gender, religion, ethnicity) are used ONLY for fairness
    monitoring and are never passed to the model.

    **Returns:** prediction, confidence, explanation, fairness report.
    """

    # ── Step 1: Load model ────────────────────────────────────────────────────
    try:
        model = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Step 2: Extract prediction features (NO sensitive attrs) ──────────────
    prediction_features = {
        "years_experience":     request.years_experience,
        "education_level":      request.education_level,
        "technical_score":      request.technical_score,
        "communication_score":  request.communication_score,
        "num_past_jobs":        request.num_past_jobs,
        "certifications":       request.certifications,
    }

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    try:
        prediction, confidence, explanation = predict(model, prediction_features)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    prediction_label = "Hired" if prediction == 1 else "Not Hired"

    # ── Step 4: Fairness check (using sensitive attrs) ────────────────────────
    # Pick the first provided sensitive attribute for the report
    sensitive_attr = None
    sensitive_value = None

    if request.gender:
        sensitive_attr, sensitive_value = "gender", request.gender
    elif request.religion:
        sensitive_attr, sensitive_value = "religion", request.religion
    elif request.ethnicity:
        sensitive_attr, sensitive_value = "ethnicity", request.ethnicity

    fairness_result = run_fairness_check(
        prediction=prediction,
        sensitive_attr=sensitive_attr or "not_provided",
        sensitive_value=sensitive_value or "unknown",
        domain="hiring",
    )

    # ── Strip sensitive value from response (privacy) ─────────────────────────
    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    # ── Step 5: Log and persist (async background task) ───────────────────────
    log_record = {
        "domain": "hiring",
        "input": prediction_features,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "explanation": explanation,
        "fairness": safe_fairness,
    }

    background_tasks.add_task(log_prediction, **{
        "domain": "hiring",
        "input_data": prediction_features,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "explanation": explanation,
        "fairness_result": fairness_result,
    })
    background_tasks.add_task(save_prediction, log_record)

    # ── Step 6: Build and return response ─────────────────────────────────────
    warning_msg = ""
    if fairness_result.get("warning"):
        warning_msg = f" ⚠️ Fairness Warning: {fairness_result['warning']}"

    return HiringResponse(
        prediction=prediction,
        prediction_label=prediction_label,
        confidence=confidence,
        explanation=explanation,
        fairness=safe_fairness,
        message=f"Prediction complete.{warning_msg}",
    )
