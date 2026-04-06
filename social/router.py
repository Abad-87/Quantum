"""
social/router.py

FastAPI router for Social Media Recommendation.
Endpoint: POST /social/recommend
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
logger = logging.getLogger("social.router")


# ─── Input schema ─────────────────────────────────────────────────────────────

class SocialRequest(BaseModel):
    avg_session_minutes: float = Field(..., ge=0,   le=1440, description="Avg daily session length (minutes)")
    posts_per_day:       float = Field(0.0, ge=0,   le=100,  description="Avg posts per day")
    topics_interacted:   int   = Field(..., ge=0,   le=50,   description="Distinct topics interacted with")
    like_rate:           float = Field(..., ge=0.0, le=1.0,  description="Ratio of liked to seen content")
    share_rate:          float = Field(0.0, ge=0.0, le=1.0,  description="Ratio of shared to liked content")
    comment_rate:        float = Field(0.0, ge=0.0, le=1.0,  description="Ratio of commented to seen content")
    account_age_days:    int   = Field(..., ge=0,   le=10000,description="Account age in days")

    # Sensitive attributes — fairness monitoring ONLY
    gender:   Optional[str] = Field(None, description="Fairness monitoring only")
    age_group: Optional[str] = Field(None, description="Fairness monitoring only")
    location:  Optional[str] = Field(None, description="Fairness monitoring only")
    language:  Optional[str] = Field(None, description="Fairness monitoring only")

    @field_validator("like_rate", "share_rate", "comment_rate")
    @classmethod
    def rate_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Rate values must be between 0.0 and 1.0")
        return v


# ─── Output schema ────────────────────────────────────────────────────────────

class SocialResponse(BaseModel):
    recommended_category_id: int
    recommended_category:    str
    confidence:              float
    shap_values:             Dict[str, float]
    shap_available:          bool
    explanation:             str
    bias_risk:               Dict[str, Any]
    fairness:                Dict[str, Any]
    model_version:           str
    model_variant:           str
    message:                 str


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/recommend", response_model=SocialResponse)
async def social_recommend(
    request:          Request,
    body:             SocialRequest,
    background_tasks: BackgroundTasks,
):
    """
    **Social Media Content Recommendation**

    Recommends a content category from behavioural signals only.
    Demographic attributes are used ONLY for fairness auditing —
    never to drive the recommendation (prevents demographic filter bubbles).

    **Returns:** category, confidence, SHAP values, bias risk, explanation,
    fairness report, model version.
    """
    correlation_id: str = getattr(request.state, "correlation_id", "unknown")

    model, variant = get_model_ab()
    model_meta     = get_metadata(variant)
    model_version  = model_meta.get("version", "unknown")

    prediction_features = {
        "avg_session_minutes": body.avg_session_minutes,
        "posts_per_day":       body.posts_per_day,
        "topics_interacted":   body.topics_interacted,
        "like_rate":           body.like_rate,
        "share_rate":          body.share_rate,
        "comment_rate":        body.comment_rate,
        "account_age_days":    body.account_age_days,
    }

    sensitive_attr = next(
        (attr for attr, val in [
            ("gender",    body.gender),
            ("age_group", body.age_group),
            ("location",  body.location),
            ("language",  body.language),
        ] if val),
        None,
    )

    try:
        result = predict(
            model,
            prediction_features,
            sensitive_attr=sensitive_attr,
            domain="social",
        )
    except Exception as exc:
        logger.error(f"[{correlation_id}] Social prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}")

    fairness_result = run_fairness_check(
        prediction=result["prediction"],
        sensitive_attr=sensitive_attr or "not_provided",
        sensitive_value=next(
            (v for v in [body.gender, body.age_group, body.location, body.language] if v),
            "unknown",
        ),
        domain="social",
    )
    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    log_record = {
        "domain":           "social",
        "input":            prediction_features,
        "prediction":       result["prediction"],
        "prediction_label": result["category_label"],
        "explanation":      result["explanation"],
        "fairness":         safe_fairness,
        "model_version":    model_version,
        "model_variant":    variant,
        "correlation_id":   correlation_id,
    }
    background_tasks.add_task(log_prediction,
        domain          = "social",
        input_data      = prediction_features,
        prediction      = result["prediction"],
        prediction_label= result["category_label"],
        explanation     = result["explanation"],
        fairness_result = fairness_result,
        correlation_id  = correlation_id,
    )
    background_tasks.add_task(save_prediction, log_record)
    background_tasks.add_task(
        log_correlation_event,
        correlation_id  = correlation_id,
        event           = "prediction_complete",
        path            = "/social/recommend",
        method          = "POST",
        model_metadata  = model_meta,
        result          = {
            "prediction":        result["prediction"],
            "prediction_label":  result["category_label"],
            "confidence":        result["confidence"],
            "bias_risk_score":   result["bias_risk"]["score"],
            "bias_risk_band":    result["bias_risk"]["band"],
        },
    )

    warning_msg = f" ⚠️ {fairness_result['warning']}" if fairness_result.get("warning") else ""

    return SocialResponse(
        recommended_category_id = result["prediction"],
        recommended_category    = result["category_label"],
        confidence              = result["confidence"],
        shap_values             = result["shap_values"],
        shap_available          = result["shap_available"],
        explanation             = result["explanation"],
        bias_risk               = result["bias_risk"],
        fairness                = safe_fairness,
        model_version           = model_version,
        model_variant           = variant,
        message                 = f"Recommendation complete.{warning_msg}",
    )
