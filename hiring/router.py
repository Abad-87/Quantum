"""
hiring/router.py

FastAPI router for the Job Hiring domain.
Endpoint: POST /hiring/predict

Changes from original
---------------------
• predict() now returns a structured dict — unpacked here instead of a tuple.
• Response schema extended with shap_values and bias_risk fields.
• correlation_id (injected by middleware) is threaded through the log call.
• model variant selected via get_model_ab() for A/B fairness testing.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, Optional
import logging

from .model_loader import get_model_ab, get_metadata
from .predictor import predict
from fairness.checker import run_fairness_check
from utils.logger import log_prediction, log_correlation_event
from utils.database import save_prediction

router = APIRouter()
logger = logging.getLogger("hiring.router")


# ─── Input schema ─────────────────────────────────────────────────────────────

class HiringRequest(BaseModel):
    # Prediction features (model sees these)
    years_experience:    float = Field(..., ge=0,   le=50,  description="Years of work experience")
    education_level:     int   = Field(..., ge=0,   le=3,   description="0=HS, 1=Bachelor, 2=Master, 3=PhD")
    technical_score:     float = Field(..., ge=0,   le=100, description="Technical assessment score")
    communication_score: float = Field(..., ge=0,   le=100, description="Communication assessment score")
    num_past_jobs:       int   = Field(..., ge=0,   le=30,  description="Number of previous jobs")
    certifications:      int   = Field(0,   ge=0,   le=20,  description="Number of relevant certifications")

    # Sensitive attributes — fairness monitoring ONLY
    gender:    Optional[str] = Field(None, description="Fairness monitoring only")
    religion:  Optional[str] = Field(None, description="Fairness monitoring only")
    ethnicity: Optional[str] = Field(None, description="Fairness monitoring only")

    @field_validator("education_level")
    @classmethod
    def validate_education(cls, v):
        if v not in {0, 1, 2, 3}:
            raise ValueError("education_level must be 0, 1, 2, or 3")
        return v


# ─── Output schema ────────────────────────────────────────────────────────────

class HiringResponse(BaseModel):
    prediction:        int
    prediction_label:  str
    confidence:        float
    shap_values:       Dict[str, float]
    shap_available:    bool
    explanation:       str
    bias_risk:         Dict[str, Any]
    fairness:          Dict[str, Any]
    model_version:     str
    model_variant:     str
    message:           str


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=HiringResponse)
async def hiring_predict(
    request:          Request,
    body:             HiringRequest,
    background_tasks: BackgroundTasks,
):
    """
    **Job Hiring Prediction**

    Predicts whether a candidate should be hired from objective features.
    Sensitive attributes (gender, religion, ethnicity) are used ONLY for
    fairness monitoring — they never reach the model.

    **Returns:** prediction, confidence, SHAP values, bias risk, explanation,
    fairness report, model version.
    """
    # ── Retrieve correlation_id set by the middleware ─────────────────────────
    correlation_id: str = getattr(request.state, "correlation_id", "unknown")

    # ── Select model via A/B router ───────────────────────────────────────────
    model, variant = get_model_ab()
    model_meta     = get_metadata(variant)
    model_version  = model_meta.get("version", "unknown")

    # ── Separate prediction features from sensitive attributes ─────────────────
    prediction_features = {
        "years_experience":     body.years_experience,
        "education_level":      body.education_level,
        "technical_score":      body.technical_score,
        "communication_score":  body.communication_score,
        "num_past_jobs":        body.num_past_jobs,
        "certifications":       body.certifications,
    }

    sensitive_attr  = next(
        (attr for attr, val in [
            ("gender", body.gender),
            ("religion", body.religion),
            ("ethnicity", body.ethnicity),
        ] if val),
        None,
    )

    # ── Run prediction (returns structured dict) ──────────────────────────────
    try:
        result = predict(
            model,
            prediction_features,
            sensitive_attr=sensitive_attr,
            domain="hiring",
        )
    except Exception as exc:
        logger.error(f"[{correlation_id}] Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    prediction_label = "Hired" if result["prediction"] == 1 else "Not Hired"

    # ── Fairness check ────────────────────────────────────────────────────────
    fairness_result = run_fairness_check(
        prediction=result["prediction"],
        sensitive_attr=sensitive_attr or "not_provided",
        sensitive_value=next(
            (val for val in [body.gender, body.religion, body.ethnicity] if val),
            "unknown",
        ),
        domain="hiring",
    )
    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    # ── Async: persist + prediction log ───────────────────────────────────────
    log_record = {
        "domain":           "hiring",
        "input":            prediction_features,
        "prediction":       result["prediction"],
        "prediction_label": prediction_label,
        "explanation":      result["explanation"],
        "fairness":         safe_fairness,
        "model_version":    model_version,
        "model_variant":    variant,
        "correlation_id":   correlation_id,
    }
    background_tasks.add_task(log_prediction,
        domain          = "hiring",
        input_data      = prediction_features,
        prediction      = result["prediction"],
        prediction_label= prediction_label,
        explanation     = result["explanation"],
        fairness_result = fairness_result,
        correlation_id  = correlation_id,
    )
    background_tasks.add_task(save_prediction, log_record)

    # ── Emit correlation audit completion event ───────────────────────────────
    background_tasks.add_task(
        log_correlation_event,
        correlation_id  = correlation_id,
        event           = "prediction_complete",
        path            = "/hiring/predict",
        method          = "POST",
        model_metadata  = model_meta,
        result          = {
            "prediction":        result["prediction"],
            "prediction_label":  prediction_label,
            "confidence":        result["confidence"],
            "bias_risk_score":   result["bias_risk"]["score"],
            "bias_risk_band":    result["bias_risk"]["band"],
        },
    )

    warning_msg = f" ⚠️ {fairness_result['warning']}" if fairness_result.get("warning") else ""

    return HiringResponse(
        prediction       = result["prediction"],
        prediction_label = prediction_label,
        confidence       = result["confidence"],
        shap_values      = result["shap_values"],
        shap_available   = result["shap_available"],
        explanation      = result["explanation"],
        bias_risk        = result["bias_risk"],
        fairness         = safe_fairness,
        model_version    = model_version,
        model_variant    = variant,
        message          = f"Prediction complete.{warning_msg}",
    )
