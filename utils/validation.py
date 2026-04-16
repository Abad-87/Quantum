"""
utils/validation.py

Centralised Pydantic v2 Validation Schemas — Phase 4: Security & Privacy
=========================================================================

All request and response models live here.  Routers import directly from
this module, keeping domain code thin and validation logic auditable in one
place.

Validation layers applied to every field
-----------------------------------------
1. Type coercion     — Pydantic converts compatible types (e.g. "5" → 5).
2. Range / length    — Field(ge=, le=, min_length=, max_length=).
3. Pattern matching  — Field(pattern=) for categorical string fields.
4. Injection guard   — @field_validator strips control characters, null bytes,
                       CRLF sequences, and common injection prefixes
                       (SQL keywords, script tags, shell metacharacters).
5. Cross-field rules — @model_validator(mode="after") enforces business logic
                       that spans multiple fields (e.g. loan_amount ≤ 10 ×
                       annual_income).
6. Normalisation     — string fields are stripped of leading/trailing whitespace
                       and lowercased where appropriate.

Sensitive-attribute string rules
---------------------------------
Sensitive attributes (gender, religion, ethnicity, age_group, location,
language) are:
  - Optional (may be omitted entirely)
  - Max 64 characters
  - Pattern: printable ASCII letters, digits, hyphens, underscores, spaces
  - Injection-guarded (no control chars, SQL fragments, script tags)
  - NOT validated against an exhaustive allowlist — this project receives
    data from many cultural contexts and an allowlist would be exclusionary.

Response models
---------------
All response models use strict=False (default) so FastAPI can serialise
Python dataclasses and dicts without extra coercion.  They do not perform
input validation — they are output contracts only.

Security error responses
------------------------
ValidationErrorResponse and RateLimitResponse are returned by the custom
exception handlers in main.py and provide a consistent error envelope that
never leaks internal stack traces.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ═════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Maximum string length for sensitive-attribute fields.
_SENSITIVE_MAX_LEN: int = 64

# Compiled injection-detection pattern.
# Matches: null bytes, CRLF injection, common SQL keywords used as values,
# script/HTML tags, shell command prefixes, and C-style escape sequences.
_INJECTION_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"    # control chars (excl. \t \n \r)
    r"|(\r\n|\r)"                             # CRLF / CR injection
    r"|(--|;|/\*|\*/)"                        # SQL comment / statement sep
    r"|(<\s*script|</\s*script|<\s*img|javascript:)",  # XSS vectors
    re.IGNORECASE,
)

# Pattern that sensitive-attribute string values must satisfy.
# Allows: letters (any Unicode letter via \w), digits, hyphen, space, underscore.
_SENSITIVE_PATTERN = re.compile(r"^[\w\s\-\+]+$", re.UNICODE)

# Valid loan term months — standard consumer lending tenors.
_VALID_LOAN_TERMS: frozenset[int] = frozenset({
    6, 12, 18, 24, 30, 36, 48, 60, 84, 120, 180, 240, 360,
})

# Age-group format: "NN-NN"  e.g. "18-24", "65+"
_AGE_GROUP_RE = re.compile(r"^\d{1,3}(?:-\d{1,3}|\+)$")

# BCP-47-like language code: "en", "en-US", "zh-Hans"
_LANGUAGE_RE = re.compile(r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{1,8})*$")


# ═════════════════════════════════════════════════════════════════════════════
# SHARED FIELD VALIDATORS
# ═════════════════════════════════════════════════════════════════════════════

def _guard_injection(value: Optional[str], field_name: str) -> Optional[str]:
    """
    Raise ValueError if *value* contains injection patterns.
    Returns the stripped value on success.
    """
    if value is None:
        return None
    stripped = value.strip()
    if _INJECTION_RE.search(stripped):
        raise ValueError(
            f"Field '{field_name}' contains disallowed characters or patterns."
        )
    return stripped


def _validate_sensitive_str(value: Optional[str], field_name: str) -> Optional[str]:
    """
    Full validation pipeline for optional sensitive-attribute strings:
      1. Injection guard
      2. Max-length check (64 chars)
      3. Character allowlist (printable word chars + space + hyphen)
    Returns normalised string or None.
    """
    if value is None or value == "":
        return None
    cleaned = _guard_injection(value, field_name)
    if len(cleaned) > _SENSITIVE_MAX_LEN:
        raise ValueError(
            f"Field '{field_name}' exceeds maximum length of {_SENSITIVE_MAX_LEN} characters."
        )
    if not _SENSITIVE_PATTERN.match(cleaned):
        raise ValueError(
            f"Field '{field_name}' contains invalid characters. "
            "Use letters, digits, spaces, hyphens, or underscores only."
        )
    return cleaned.lower()   # normalise to lowercase for consistent grouping


# ═════════════════════════════════════════════════════════════════════════════
# HIRING SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class HiringRequest(BaseModel):
    """
    Input schema for POST /hiring/predict.

    Validation rules applied beyond type coercion
    ----------------------------------------------
    years_experience   : 0 – 50  (no fractional year below 0)
    education_level    : 0 | 1 | 2 | 3  (enum-style; field_validator enforces)
    technical_score    : 0.0 – 100.0  (2 decimal precision enforced)
    communication_score: 0.0 – 100.0
    num_past_jobs      : 0 – 30
    certifications     : 0 – 20  (default 0)
    Cross-field        : technical_score + communication_score must individually
                         be ≥ 0; combined they must not both be exactly 0 on
                         a non-zero-experience candidate (data quality check).
    Sensitive attrs    : injection-guarded, max 64 chars, word-chars only.
    """

    # ── Prediction features ───────────────────────────────────────────────────
    years_experience: float = Field(
        ..., ge=0.0, le=50.0,
        description="Years of relevant work experience (0–50).",
        examples=[5.0],
    )
    education_level: int = Field(
        ..., ge=0, le=3,
        description="0=High School, 1=Bachelor's, 2=Master's, 3=PhD.",
        examples=[2],
    )
    technical_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Technical assessment score (0–100).",
        examples=[82.0],
    )
    communication_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Communication assessment score (0–100).",
        examples=[75.0],
    )
    num_past_jobs: int = Field(
        ..., ge=0, le=30,
        description="Number of previous positions held (0–30).",
        examples=[3],
    )
    certifications: int = Field(
        default=0, ge=0, le=20,
        description="Number of relevant professional certifications (0–20).",
        examples=[2],
    )

    # ── Sensitive attributes (fairness monitoring ONLY) ───────────────────────
    gender:    Optional[str] = Field(None, description="Fairness monitoring only — not used in prediction.")
    religion:  Optional[str] = Field(None, description="Fairness monitoring only — not used in prediction.")
    ethnicity: Optional[str] = Field(None, description="Fairness monitoring only — not used in prediction.")

    # ── Field validators ──────────────────────────────────────────────────────

    @field_validator("education_level")
    @classmethod
    def education_must_be_valid(cls, v: int) -> int:
        if v not in {0, 1, 2, 3}:
            raise ValueError(
                "education_level must be one of: 0 (High School), "
                "1 (Bachelor's), 2 (Master's), 3 (PhD)."
            )
        return v

    @field_validator("technical_score", "communication_score")
    @classmethod
    def scores_two_decimal(cls, v: float) -> float:
        # Round to 2 decimal places — prevents floating-point representation attacks
        return round(v, 2)

    @field_validator("gender", "religion", "ethnicity")
    @classmethod
    def validate_sensitive(cls, v: Optional[str], info) -> Optional[str]:
        return _validate_sensitive_str(v, info.field_name)

    # ── Cross-field validation ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def cross_field_checks(self) -> "HiringRequest":
        # Data quality: if candidate has experience, at least one score > 0
        if self.years_experience > 0:
            if self.technical_score == 0.0 and self.communication_score == 0.0:
                raise ValueError(
                    "A candidate with work experience must have at least one "
                    "non-zero assessment score (technical_score or communication_score)."
                )
        return self

    model_config = {"str_strip_whitespace": True, "extra": "forbid"}


class HiringResponse(BaseModel):
    prediction:        int
    prediction_label:  str
    confidence:        float
    shap_values:       Dict[str, float]
    shap_available:    bool
    shap_status:       str
    shap_poll_url:     str
    explanation:       str
    bias_risk:         Dict[str, Any]
    fairness:          Dict[str, Any]
    preprocessing:     Dict[str, Any]
    model_version:     str
    model_variant:     str
    correlation_id:    str
    message:           str


# ═════════════════════════════════════════════════════════════════════════════
# LOAN SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class LoanRequest(BaseModel):
    """
    Input schema for POST /loan/predict.

    Validation rules applied beyond type coercion
    ----------------------------------------------
    credit_score       : 300 – 850  (FICO range)
    annual_income      : 0 – 10,000,000  (upper cap prevents absurd values)
    loan_amount        : 100 – 5,000,000
    loan_term_months   : must be one of the standard tenors
                         {6,12,18,24,30,36,48,60,84,120,180,240,360}
    employment_years   : 0 – 50
    existing_debt      : 0 – 5,000,000
    num_credit_lines   : 0 – 50
    Cross-field        : loan_amount must not exceed 10× annual_income
                         (prevents obviously fraudulent applications that
                          would fail underwriting anyway — faster rejection
                          with a meaningful error message).
    age_group          : must match "NN-NN" or "NN+" pattern.
    """

    # ── Prediction features ───────────────────────────────────────────────────
    credit_score: int = Field(
        ..., ge=300, le=850,
        description="FICO credit score (300–850).",
        examples=[720],
    )
    annual_income: float = Field(
        ..., ge=0.0, le=10_000_000.0,
        description="Gross annual income in USD (0 – $10,000,000).",
        examples=[75000.0],
    )
    loan_amount: float = Field(
        ..., ge=100.0, le=5_000_000.0,
        description="Requested loan principal in USD ($100 – $5,000,000).",
        examples=[25000.0],
    )
    loan_term_months: int = Field(
        ..., ge=6, le=360,
        description=(
            "Repayment term in months. "
            "Allowed: 6,12,18,24,30,36,48,60,84,120,180,240,360."
        ),
        examples=[36],
    )
    employment_years: float = Field(
        ..., ge=0.0, le=50.0,
        description="Years at current employer (0–50).",
        examples=[4.0],
    )
    existing_debt: float = Field(
        default=0.0, ge=0.0, le=5_000_000.0,
        description="Total existing debt in USD (0 – $5,000,000).",
        examples=[8000.0],
    )
    num_credit_lines: int = Field(
        default=0, ge=0, le=50,
        description="Number of open credit accounts (0–50).",
        examples=[3],
    )

    # ── Sensitive attributes (fairness monitoring ONLY) ───────────────────────
    gender:    Optional[str] = Field(None, description="Fairness monitoring only.")
    religion:  Optional[str] = Field(None, description="Fairness monitoring only.")
    ethnicity: Optional[str] = Field(None, description="Fairness monitoring only.")
    age_group: Optional[str] = Field(
        None,
        description=(
            "Age bracket for fairness monitoring only. "
            "Format: 'NN-NN' or 'NN+', e.g. '26-40', '65+'."
        ),
    )

    # ── Field validators ──────────────────────────────────────────────────────

    @field_validator("loan_amount")
    @classmethod
    def loan_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("loan_amount must be greater than 0.")
        return round(v, 2)

    @field_validator("annual_income", "existing_debt")
    @classmethod
    def currency_two_decimal(cls, v: float) -> float:
        return round(v, 2)

    @field_validator("loan_term_months")
    @classmethod
    def term_must_be_standard(cls, v: int) -> int:
        if v not in _VALID_LOAN_TERMS:
            raise ValueError(
                f"loan_term_months must be one of the standard tenors: "
                f"{sorted(_VALID_LOAN_TERMS)}."
            )
        return v

    @field_validator("gender", "religion", "ethnicity")
    @classmethod
    def validate_sensitive_basic(cls, v: Optional[str], info) -> Optional[str]:
        return _validate_sensitive_str(v, info.field_name)

    @field_validator("age_group")
    @classmethod
    def validate_age_group(cls, v: Optional[str]) -> Optional[str]:
        cleaned = _validate_sensitive_str(v, "age_group")
        if cleaned and not _AGE_GROUP_RE.match(cleaned.replace(" ", "")):
            raise ValueError(
                "age_group must match the format 'NN-NN' or 'NN+', "
                "e.g. '26-40', '65+'."
            )
        return cleaned

    # ── Cross-field validation ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def cross_field_checks(self) -> "LoanRequest":
        # Loan-to-income sanity check
        if self.annual_income > 0 and self.loan_amount > self.annual_income * 10:
            raise ValueError(
                f"loan_amount (${self.loan_amount:,.2f}) exceeds 10× annual_income "
                f"(${self.annual_income:,.2f}). This application cannot proceed."
            )
        return self

    model_config = {"str_strip_whitespace": True, "extra": "forbid"}


class LoanResponse(BaseModel):
    prediction:       int
    prediction_label: str
    confidence:       float
    shap_values:      Dict[str, float]
    shap_available:   bool
    shap_status:      str
    shap_poll_url:    str
    explanation:      str
    bias_risk:        Dict[str, Any]
    fairness:         Dict[str, Any]
    preprocessing:    Dict[str, Any]
    model_version:    str
    model_variant:    str
    correlation_id:   str
    message:          str


# ═════════════════════════════════════════════════════════════════════════════
# SOCIAL SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class SocialRequest(BaseModel):
    """
    Input schema for POST /social/recommend.

    Validation rules applied beyond type coercion
    ----------------------------------------------
    avg_session_minutes: 0 – 1440  (max 24 h in minutes)
    posts_per_day       : 0 – 100
    topics_interacted   : 0 – 50
    like_rate           : 0.0 – 1.0  (proportion)
    share_rate          : 0.0 – 1.0
    comment_rate        : 0.0 – 1.0
    account_age_days    : 0 – 10000 (~27 years)
    Cross-field         : like_rate + share_rate + comment_rate must each
                          be ≤ 1.0 and the sum must be ≤ 3.0 — enforced by
                          Field constraints; no cross-field needed here.
                          However: share_rate ≤ like_rate is a reasonable
                          business rule (you can only share what you liked).
    location            : injection-guarded, max 64 chars, printable only.
    language            : must match BCP-47-like pattern (e.g. "en", "en-US").
    age_group           : must match "NN-NN" or "NN+" pattern.
    """

    # ── Prediction features ───────────────────────────────────────────────────
    avg_session_minutes: float = Field(
        ..., ge=0.0, le=1440.0,
        description="Average daily session length in minutes (0–1440).",
        examples=[45.0],
    )
    posts_per_day: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Average content posts per day (0–100).",
        examples=[3.0],
    )
    topics_interacted: int = Field(
        ..., ge=0, le=50,
        description="Number of distinct content topics interacted with (0–50).",
        examples=[12],
    )
    like_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Proportion of seen content that was liked (0.0–1.0).",
        examples=[0.65],
    )
    share_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Proportion of liked content that was shared (0.0–1.0).",
        examples=[0.2],
    )
    comment_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Proportion of seen content that received a comment (0.0–1.0).",
        examples=[0.1],
    )
    account_age_days: int = Field(
        ..., ge=0, le=10000,
        description="Account age in days (0–10,000 ≈ 27 years).",
        examples=[365],
    )

    # ── Sensitive attributes (fairness monitoring ONLY) ───────────────────────
    gender:    Optional[str] = Field(None, description="Fairness monitoring only.")
    age_group: Optional[str] = Field(
        None,
        description="Age bracket for fairness monitoring. Format: 'NN-NN' or 'NN+'.",
    )
    location:  Optional[str] = Field(
        None,
        description="Country or region for fairness monitoring (injection-guarded, max 64 chars).",
    )
    language:  Optional[str] = Field(
        None,
        description="BCP-47 language code for fairness monitoring, e.g. 'en', 'en-US'.",
    )

    # ── Field validators ──────────────────────────────────────────────────────

    @field_validator("like_rate", "share_rate", "comment_rate")
    @classmethod
    def rate_four_decimal(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Rate values must be between 0.0 and 1.0.")
        return round(v, 4)

    @field_validator("avg_session_minutes", "posts_per_day")
    @classmethod
    def non_negative_float(cls, v: float) -> float:
        return round(v, 4)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        return _validate_sensitive_str(v, "gender")

    @field_validator("age_group")
    @classmethod
    def validate_age_group(cls, v: Optional[str]) -> Optional[str]:
        cleaned = _validate_sensitive_str(v, "age_group")
        if cleaned and not _AGE_GROUP_RE.match(cleaned.replace(" ", "")):
            raise ValueError(
                "age_group must match 'NN-NN' or 'NN+', e.g. '25-34', '65+'."
            )
        return cleaned

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: Optional[str]) -> Optional[str]:
        """
        Location is injection-guarded but NOT validated against an allowlist —
        country/region names vary too widely to enumerate safely.
        """
        return _validate_sensitive_str(v, "location")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        cleaned = _guard_injection(v.strip(), "language")
        if not _LANGUAGE_RE.match(cleaned):
            raise ValueError(
                "language must be a BCP-47 language code such as 'en', 'en-US', or 'zh-Hans'."
            )
        return cleaned.lower()

    # ── Cross-field validation ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def cross_field_checks(self) -> "SocialRequest":
        # share_rate ≤ like_rate: you can't share more than you liked
        if self.share_rate > self.like_rate + 1e-6:
            raise ValueError(
                f"share_rate ({self.share_rate}) cannot exceed like_rate ({self.like_rate}). "
                "You can only share content you have already liked."
            )
        return self

    model_config = {"str_strip_whitespace": True, "extra": "forbid"}


class SocialResponse(BaseModel):
    recommended_category_id: int
    recommended_category:    str
    confidence:              float
    shap_values:             Dict[str, float]
    shap_available:          bool
    shap_status:             str
    shap_poll_url:           str
    explanation:             str
    bias_risk:               Dict[str, Any]
    fairness:                Dict[str, Any]
    preprocessing:           Dict[str, Any]
    model_version:           str
    model_variant:           str
    correlation_id:          str
    message:                 str


# ═════════════════════════════════════════════════════════════════════════════
# ERROR RESPONSE SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class ValidationErrorDetail(BaseModel):
    """Single field-level validation failure."""
    field:   str
    message: str
    input:   Optional[Any] = None


class ValidationErrorResponse(BaseModel):
    """
    Standard 422 response body.  Never exposes internal stack traces.
    Returned by the custom validation exception handler in main.py.
    """
    error:          str = "Validation failed"
    correlation_id: Optional[str] = None
    details:        List[ValidationErrorDetail] = []


class RateLimitResponse(BaseModel):
    """Standard 429 response body."""
    error:          str = "Rate limit exceeded"
    retry_after_s:  int
    correlation_id: Optional[str] = None


class SecurityErrorResponse(BaseModel):
    """Standard 400 response body for injection / malformed-request rejections."""
    error:          str = "Request rejected"
    reason:         str
    correlation_id: Optional[str] = None
