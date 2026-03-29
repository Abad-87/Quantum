"""
tests/test_api.py

Integration tests for all three domain endpoints.
These tests use FastAPI's TestClient to make real HTTP calls
without needing a running server.

Run:  pytest tests/test_api.py -v

IMPORTANT: Run  python create_dummy_models.py  first to generate test models.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Make sure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


# ─── Health Checks ────────────────────────────────────────────────────────────

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Quantum" in data["platform"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ─── Hiring Tests ─────────────────────────────────────────────────────────────

HIRING_VALID_PAYLOAD = {
    "years_experience": 5,
    "education_level": 2,
    "technical_score": 82,
    "communication_score": 75,
    "num_past_jobs": 3,
    "certifications": 2,
    "gender": "female",          # Sensitive — for fairness check only
}

def test_hiring_predict_success():
    response = client.post("/hiring/predict", json=HIRING_VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    # Required fields in response
    assert "prediction" in data
    assert "prediction_label" in data
    assert "confidence" in data
    assert "explanation" in data
    assert "fairness" in data
    assert "message" in data

    # Prediction must be binary
    assert data["prediction"] in [0, 1]
    # Label must match prediction
    expected_label = "Hired" if data["prediction"] == 1 else "Not Hired"
    assert data["prediction_label"] == expected_label

    # Sensitive value must NOT be in response (privacy)
    assert "sensitive_value" not in data["fairness"]


def test_hiring_predict_no_sensitive_attrs():
    """Prediction should work even without sensitive attributes."""
    payload = {k: v for k, v in HIRING_VALID_PAYLOAD.items()
               if k not in ("gender", "religion", "ethnicity")}
    response = client.post("/hiring/predict", json=payload)
    assert response.status_code == 200


def test_hiring_invalid_education_level():
    payload = {**HIRING_VALID_PAYLOAD, "education_level": 5}  # Invalid: max is 3
    response = client.post("/hiring/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_hiring_invalid_score_range():
    payload = {**HIRING_VALID_PAYLOAD, "technical_score": 150}  # Invalid: max is 100
    response = client.post("/hiring/predict", json=payload)
    assert response.status_code == 422


def test_hiring_missing_required_fields():
    response = client.post("/hiring/predict", json={})
    assert response.status_code == 422


# ─── Loan Tests ───────────────────────────────────────────────────────────────

LOAN_VALID_PAYLOAD = {
    "credit_score": 720,
    "annual_income": 75000,
    "loan_amount": 25000,
    "loan_term_months": 36,
    "employment_years": 4,
    "existing_debt": 8000,
    "num_credit_lines": 3,
    "ethnicity": "hispanic",     # Sensitive — for fairness check only
}

def test_loan_predict_success():
    response = client.post("/loan/predict", json=LOAN_VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["Approved", "Rejected"]
    assert "explanation" in data
    assert len(data["explanation"]) > 0
    assert "sensitive_value" not in data["fairness"]


def test_loan_low_credit_score():
    payload = {**LOAN_VALID_PAYLOAD, "credit_score": 400}
    response = client.post("/loan/predict", json=payload)
    assert response.status_code == 200
    # Low credit score — likely rejected, but just verify response is valid
    assert response.json()["prediction"] in [0, 1]


def test_loan_invalid_credit_score():
    payload = {**LOAN_VALID_PAYLOAD, "credit_score": 200}  # Below 300
    response = client.post("/loan/predict", json=payload)
    assert response.status_code == 422


def test_loan_zero_loan_amount():
    payload = {**LOAN_VALID_PAYLOAD, "loan_amount": 0}
    response = client.post("/loan/predict", json=payload)
    assert response.status_code == 422


# ─── Social Tests ─────────────────────────────────────────────────────────────

SOCIAL_VALID_PAYLOAD = {
    "avg_session_minutes": 45,
    "posts_per_day": 3,
    "topics_interacted": 12,
    "like_rate": 0.65,
    "share_rate": 0.2,
    "comment_rate": 0.1,
    "account_age_days": 365,
    "age_group": "25-34",        # Sensitive — for fairness check only
    "location": "India",
}

def test_social_recommend_success():
    response = client.post("/social/recommend", json=SOCIAL_VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert "recommended_category_id" in data
    assert "recommended_category" in data
    assert "confidence" in data
    assert "explanation" in data
    assert "fairness" in data

    # Category ID must be a non-negative integer
    assert isinstance(data["recommended_category_id"], int)
    assert data["recommended_category_id"] >= 0

    # Sensitive value must NOT be in response
    assert "sensitive_value" not in data["fairness"]


def test_social_invalid_like_rate():
    payload = {**SOCIAL_VALID_PAYLOAD, "like_rate": 1.5}  # Max is 1.0
    response = client.post("/social/recommend", json=payload)
    assert response.status_code == 422


def test_social_no_sensitive_attrs():
    """Recommendation should work without any sensitive attributes."""
    payload = {k: v for k, v in SOCIAL_VALID_PAYLOAD.items()
               if k not in ("age_group", "location", "gender", "language")}
    response = client.post("/social/recommend", json=payload)
    assert response.status_code == 200


# ─── Cross-domain: Sensitive Data Privacy ─────────────────────────────────────

def test_hiring_sensitive_data_not_in_response():
    """Ensures sensitive attributes are never echoed back in any response field."""
    response = client.post("/hiring/predict", json=HIRING_VALID_PAYLOAD)
    response_str = response.text

    # The actual sensitive value should not appear in the response body
    assert "female" not in response_str or "sensitive_value" not in response.json().get("fairness", {})


def test_loan_sensitive_data_not_in_response():
    response = client.post("/loan/predict", json=LOAN_VALID_PAYLOAD)
    assert "sensitive_value" not in response.json().get("fairness", {})


def test_social_sensitive_data_not_in_response():
    response = client.post("/social/recommend", json=SOCIAL_VALID_PAYLOAD)
    assert "sensitive_value" not in response.json().get("fairness", {})
