"""
social/router.py
FastAPI router for Social Media Recommendation.
Endpoint: POST /social/recommend
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
logger = logging.getLogger("social.router")


# ─── Input Schema ─────────────────────────────────────────────────────────────
class SocialRequest(BaseModel):
    # ── Prediction features (behavioral — no demographics) ────────────────────
    avg_session_minutes: float = Field(..., ge=0,   le=1440, description="Average daily session length in minutes")
    posts_per_day: float       = Field(0.0, ge=0,   le=100,  description="Average posts per day")
    topics_interacted: int     = Field(..., ge=0,   le=50,   description="Number of distinct topics interacted with")
    like_rate: float           = Field(..., ge=0.0, le=1.0,  description="Ratio of liked to seen content")
    share_rate: float          = Field(0.0, ge=0.0, le=1.0,  description="Ratio of shared to liked content")
    comment_rate: float        = Field(0.0, ge=0.0, le=1.0,  description="Ratio of commented to seen content")
    account_age_days: int      = Field(..., ge=0,   le=10000, description="Account age in days")

    # ── Sensitive attributes (fairness monitoring only) ───────────────────────
    gender: str | None      = Field(None, description="Fairness monitoring only")
    age_group: str | None   = Field(None, description="Fairness monitoring only (e.g. '18-24')")
    location: str | None    = Field(None, description="Fairness monitoring only (country/region)")
    language: str | None    = Field(None, description="Fairness monitoring only")

    @field_validator("like_rate", "share_rate", "comment_rate")
    @classmethod
    def rate_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Rate values must be between 0.0 and 1.0")
        return v


# ─── Output Schema ────────────────────────────────────────────────────────────
class SocialResponse(BaseModel):
    recommended_category_id: int
    recommended_category: str
    confidence: float
    explanation: str
    fairness: dict
    message: str


# ─── Endpoint ─────────────────────────────────────────────────────────────────
@router.post("/recommend", response_model=SocialResponse)
async def social_recommend(request: SocialRequest, background_tasks: BackgroundTasks):
    """
    **Social Media Content Recommendation**

    Recommends a content category based on behavioral signals only.
    Demographic attributes (gender, age, location, language) are used
    ONLY for fairness auditing — never to make the recommendation.

    This prevents filter bubbles driven by demographic profiling.

    **Returns:** recommended category, confidence, explanation, fairness report.
    """

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Extract prediction features ───────────────────────────────────────────
    prediction_features = {
        "avg_session_minutes": request.avg_session_minutes,
        "posts_per_day":       request.posts_per_day,
        "topics_interacted":   request.topics_interacted,
        "like_rate":           request.like_rate,
        "share_rate":          request.share_rate,
        "comment_rate":        request.comment_rate,
        "account_age_days":    request.account_age_days,
    }

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        category_id, category_label, confidence, explanation = predict(model, prediction_features)
    except Exception as e:
        logger.error(f"Social prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

    # ── Fairness check ────────────────────────────────────────────────────────
    sensitive_attr  = None
    sensitive_value = None

    for attr, val in [
        ("gender",    request.gender),
        ("age_group", request.age_group),
        ("location",  request.location),
        ("language",  request.language),
    ]:
        if val:
            sensitive_attr, sensitive_value = attr, val
            break

    fairness_result = run_fairness_check(
        prediction=category_id,
        sensitive_attr=sensitive_attr or "not_provided",
        sensitive_value=sensitive_value or "unknown",
        domain="social",
    )

    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    # ── Log & persist ─────────────────────────────────────────────────────────
    log_record = {
        "domain": "social",
        "input": prediction_features,
        "prediction": category_id,
        "prediction_label": category_label,
        "explanation": explanation,
        "fairness": safe_fairness,
    }
    background_tasks.add_task(log_prediction, **{
        "domain": "social",
        "input_data": prediction_features,
        "prediction": category_id,
        "prediction_label": category_label,
        "explanation": explanation,
        "fairness_result": fairness_result,
    })
    background_tasks.add_task(save_prediction, log_record)

    # ── Response ──────────────────────────────────────────────────────────────
    warning_msg = ""
    if fairness_result.get("warning"):
        warning_msg = f" ⚠️ Fairness Warning: {fairness_result['warning']}"

    return SocialResponse(
        recommended_category_id=category_id,
        recommended_category=category_label,
        confidence=confidence,
        explanation=explanation,
        fairness=safe_fairness,
        message=f"Recommendation complete.{warning_msg}",
    )
