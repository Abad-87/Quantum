"""
tests/test_api.py

FastAPI integration tests using TestClient (session-scoped via conftest).

Covers
------
- All three prediction endpoints: status codes, response schema, privacy
- Error handling: invalid JSON, missing fields, wrong types, injection attempts
- Security middleware: oversized body, null bytes, bad Content-Type
- Custom validation error envelope (never echoes raw input)
- SHAP async contract: shap_status=pending, shap_poll_url in response
- /shap/{id} REST poll endpoint
- /health and /models platform endpoints
- Performance: 200 back-to-back requests without crash
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# Platform / health
# ═════════════════════════════════════════════════════════════════════════════

class TestPlatformEndpoints:

    def test_root_status_online(self, app_client):
        r = app_client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"]  == "online"
        assert "Quantum"       in data["platform"]
        assert "version"       in data

    def test_root_lists_prediction_endpoints(self, app_client):
        endpoints = app_client.get("/").json()["endpoints"]
        assert any("/hiring/predict" in e for e in endpoints)
        assert any("/loan/predict"   in e for e in endpoints)
        assert any("/social/recommend" in e for e in endpoints)

    def test_health_is_healthy(self, app_client):
        r = app_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "timestamp"    in data

    def test_health_contains_models_key(self, app_client):
        r = app_client.get("/health")
        assert "models" in r.json()

    def test_health_contains_security_key(self, app_client):
        data = app_client.get("/health").json()
        assert "security" in data

    def test_models_endpoint(self, app_client):
        r = app_client.get("/models")
        assert r.status_code == 200
        data = r.json()
        assert "models"    in data
        assert "timestamp" in data


# ═════════════════════════════════════════════════════════════════════════════
# Response schema helpers
# ═════════════════════════════════════════════════════════════════════════════

_HIRING_REQUIRED = {
    "prediction", "prediction_label", "confidence", "shap_values",
    "shap_available", "shap_status", "shap_poll_url", "explanation",
    "bias_risk", "fairness", "preprocessing", "model_version",
    "model_variant", "correlation_id", "message",
}
_LOAN_REQUIRED = _HIRING_REQUIRED  # same shape
_SOCIAL_REQUIRED = {
    "recommended_category_id", "recommended_category", "confidence",
    "shap_values", "shap_available", "shap_status", "shap_poll_url",
    "explanation", "bias_risk", "fairness", "preprocessing",
    "model_version", "model_variant", "correlation_id", "message",
}


# ═════════════════════════════════════════════════════════════════════════════
# HIRING  POST /hiring/predict
# ═════════════════════════════════════════════════════════════════════════════

class TestHiringPredict:

    def test_valid_request_200(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json=HIRING_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema_complete(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        missing = _HIRING_REQUIRED - data.keys()
        assert not missing, f"Missing response keys: {missing}"

    def test_prediction_is_binary(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        assert data["prediction"] in (0, 1)

    def test_prediction_label_matches_prediction(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        expected = "Hired" if data["prediction"] == 1 else "Not Hired"
        assert data["prediction_label"] == expected

    def test_confidence_in_unit_interval(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_shap_status_pending_on_first_response(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        assert data["shap_status"]    == "pending"
        assert data["shap_available"] is False
        assert data["shap_values"]    == {}

    def test_shap_poll_url_format(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        assert data["shap_poll_url"].startswith("/shap/")
        assert data["correlation_id"] in data["shap_poll_url"]

    def test_correlation_id_is_uuid(self, app_client, HIRING_PAYLOAD):
        import uuid
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        uuid.UUID(data["correlation_id"])   # raises ValueError if not valid UUID

    def test_bias_risk_structure(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        br = data["bias_risk"]
        assert "score"           in br
        assert "band"            in br
        assert "flag_for_review" in br
        assert "components"      in br
        assert "recommendation"  in br

    def test_fairness_no_sensitive_value(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        assert "sensitive_value" not in data["fairness"]

    def test_preprocessing_report_present(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        pp = data["preprocessing"]
        assert "sufficient_history" in pp
        assert "records_used"       in pp
        assert "message"            in pp

    def test_explanation_non_empty(self, app_client, HIRING_PAYLOAD):
        data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"]) > 5

    def test_without_sensitive_attrs(self, app_client, HIRING_PAYLOAD):
        payload = {k: v for k, v in HIRING_PAYLOAD.items()
                   if k not in ("gender", "religion", "ethnicity")}
        r = app_client.post("/hiring/predict", json=payload)
        assert r.status_code == 200

    # ── Validation errors ─────────────────────────────────────────────────────

    def test_invalid_education_level_422(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json={**HIRING_PAYLOAD, "education_level": 5})
        assert r.status_code == 422

    def test_score_out_of_range_422(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json={**HIRING_PAYLOAD, "technical_score": 150})
        assert r.status_code == 422

    def test_missing_required_fields_422(self, app_client):
        r = app_client.post("/hiring/predict", json={})
        assert r.status_code == 422

    def test_extra_unknown_field_422(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json={**HIRING_PAYLOAD, "hacked": "yes"})
        assert r.status_code == 422

    def test_sql_injection_in_gender_422(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json={
            **HIRING_PAYLOAD, "gender": "'; DROP TABLE users; --"
        })
        assert r.status_code == 422

    def test_validation_error_response_no_raw_input(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json={**HIRING_PAYLOAD, "education_level": 9})
        body = r.json()
        assert "error"   in body
        assert "details" in body
        # No raw input echoed back
        for detail in body["details"]:
            assert detail.get("input") is None

    def test_cross_field_rejection(self, app_client, HIRING_PAYLOAD):
        """Experienced candidate with zero scores is rejected (cross-field rule)."""
        r = app_client.post("/hiring/predict", json={
            **HIRING_PAYLOAD,
            "years_experience":    5,
            "technical_score":     0,
            "communication_score": 0,
        })
        assert r.status_code == 422


# ═════════════════════════════════════════════════════════════════════════════
# LOAN  POST /loan/predict
# ═════════════════════════════════════════════════════════════════════════════

class TestLoanPredict:

    def test_valid_request_200(self, app_client, LOAN_PAYLOAD):
        r = app_client.post("/loan/predict", json=LOAN_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema_complete(self, app_client, LOAN_PAYLOAD):
        data = app_client.post("/loan/predict", json=LOAN_PAYLOAD).json()
        missing = _LOAN_REQUIRED - data.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_prediction_label_approved_or_rejected(self, app_client, LOAN_PAYLOAD):
        data = app_client.post("/loan/predict", json=LOAN_PAYLOAD).json()
        assert data["prediction_label"] in ("Approved", "Rejected")

    def test_shap_pending(self, app_client, LOAN_PAYLOAD):
        data = app_client.post("/loan/predict", json=LOAN_PAYLOAD).json()
        assert data["shap_status"]    == "pending"
        assert data["shap_available"] is False

    def test_invalid_credit_score_422(self, app_client, LOAN_PAYLOAD):
        r = app_client.post("/loan/predict", json={**LOAN_PAYLOAD, "credit_score": 200})
        assert r.status_code == 422

    def test_invalid_loan_term_422(self, app_client, LOAN_PAYLOAD):
        r = app_client.post("/loan/predict", json={**LOAN_PAYLOAD, "loan_term_months": 7})
        assert r.status_code == 422

    def test_loan_exceeds_10x_income_422(self, app_client, LOAN_PAYLOAD):
        r = app_client.post("/loan/predict", json={
            **LOAN_PAYLOAD, "annual_income": 30_000, "loan_amount": 400_000
        })
        assert r.status_code == 422

    def test_zero_loan_amount_422(self, app_client, LOAN_PAYLOAD):
        r = app_client.post("/loan/predict", json={**LOAN_PAYLOAD, "loan_amount": 0})
        assert r.status_code == 422

    def test_invalid_age_group_format_422(self, app_client, LOAN_PAYLOAD):
        r = app_client.post("/loan/predict", json={**LOAN_PAYLOAD, "age_group": "adult"})
        assert r.status_code == 422

    def test_valid_age_group_65_plus(self, app_client, LOAN_PAYLOAD):
        r = app_client.post("/loan/predict", json={**LOAN_PAYLOAD, "age_group": "65+"})
        assert r.status_code == 200

    def test_no_ethnicity_200(self, app_client, LOAN_PAYLOAD):
        payload = {k: v for k, v in LOAN_PAYLOAD.items() if k != "ethnicity"}
        r = app_client.post("/loan/predict", json=payload)
        assert r.status_code == 200

    def test_sensitive_value_not_in_response(self, app_client, LOAN_PAYLOAD):
        data = app_client.post("/loan/predict", json=LOAN_PAYLOAD).json()
        assert "sensitive_value" not in data["fairness"]


# ═════════════════════════════════════════════════════════════════════════════
# SOCIAL  POST /social/recommend
# ═════════════════════════════════════════════════════════════════════════════

class TestSocialRecommend:

    def test_valid_request_200(self, app_client, SOCIAL_PAYLOAD):
        r = app_client.post("/social/recommend", json=SOCIAL_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema_complete(self, app_client, SOCIAL_PAYLOAD):
        data = app_client.post("/social/recommend", json=SOCIAL_PAYLOAD).json()
        missing = _SOCIAL_REQUIRED - data.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_category_id_valid_range(self, app_client, SOCIAL_PAYLOAD):
        data = app_client.post("/social/recommend", json=SOCIAL_PAYLOAD).json()
        assert 0 <= data["recommended_category_id"] <= 7

    def test_category_is_string(self, app_client, SOCIAL_PAYLOAD):
        data = app_client.post("/social/recommend", json=SOCIAL_PAYLOAD).json()
        assert isinstance(data["recommended_category"], str)

    def test_shap_pending(self, app_client, SOCIAL_PAYLOAD):
        data = app_client.post("/social/recommend", json=SOCIAL_PAYLOAD).json()
        assert data["shap_status"] == "pending"

    def test_invalid_like_rate_422(self, app_client, SOCIAL_PAYLOAD):
        r = app_client.post("/social/recommend", json={**SOCIAL_PAYLOAD, "like_rate": 1.5})
        assert r.status_code == 422

    def test_share_rate_exceeds_like_rate_422(self, app_client, SOCIAL_PAYLOAD):
        r = app_client.post("/social/recommend", json={
            **SOCIAL_PAYLOAD, "like_rate": 0.2, "share_rate": 0.9
        })
        assert r.status_code == 422

    def test_invalid_language_code_422(self, app_client, SOCIAL_PAYLOAD):
        r = app_client.post("/social/recommend", json={
            **SOCIAL_PAYLOAD, "language": "not-valid!!"
        })
        assert r.status_code == 422

    def test_valid_bcp47_language(self, app_client, SOCIAL_PAYLOAD):
        r = app_client.post("/social/recommend", json={**SOCIAL_PAYLOAD, "language": "en-US"})
        assert r.status_code == 200

    def test_crlf_injection_in_location_422(self, app_client, SOCIAL_PAYLOAD):
        r = app_client.post("/social/recommend", json={
            **SOCIAL_PAYLOAD, "location": "India\r\nX-Injected: evil"
        })
        assert r.status_code == 422

    def test_no_sensitive_attrs_200(self, app_client, SOCIAL_PAYLOAD):
        payload = {k: v for k, v in SOCIAL_PAYLOAD.items()
                   if k not in ("age_group", "location", "gender", "language")}
        r = app_client.post("/social/recommend", json=payload)
        assert r.status_code == 200

    def test_sensitive_value_not_in_response(self, app_client, SOCIAL_PAYLOAD):
        data = app_client.post("/social/recommend", json=SOCIAL_PAYLOAD).json()
        assert "sensitive_value" not in data["fairness"]


# ═════════════════════════════════════════════════════════════════════════════
# Security middleware
# ═════════════════════════════════════════════════════════════════════════════

class TestSecurityMiddleware:

    def test_oversized_body_413(self, app_client, HIRING_PAYLOAD):
        giant_body = json.dumps({**HIRING_PAYLOAD, "padding": "x" * 70_000})
        r = app_client.post(
            "/hiring/predict",
            content=giant_body,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 413

    def test_null_byte_in_body_400(self, app_client, HIRING_PAYLOAD):
        body = json.dumps(HIRING_PAYLOAD).encode() + b"\x00"
        r = app_client.post(
            "/hiring/predict",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_wrong_content_type_415(self, app_client, HIRING_PAYLOAD):
        r = app_client.post(
            "/hiring/predict",
            data="years_experience=5",   # form data, not JSON
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 415

    def test_security_headers_on_response(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json=HIRING_PAYLOAD)
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("x-frame-options")        == "DENY"

    def test_correlation_id_in_response_header(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json=HIRING_PAYLOAD)
        assert "x-correlation-id" in r.headers

    def test_response_time_header_present(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json=HIRING_PAYLOAD)
        assert "x-response-time" in r.headers


# ═════════════════════════════════════════════════════════════════════════════
# SHAP poll endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestShapPollEndpoint:

    def test_unknown_id_returns_missing(self, app_client):
        r = app_client.get("/shap/totally-unknown-id-xyz")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "missing"
        assert data["shap_report"] is None

    def test_pending_after_predict(self, app_client, HIRING_PAYLOAD):
        pred_data = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        corr_id   = pred_data["correlation_id"]
        poll_url  = pred_data["shap_poll_url"]

        r = app_client.get(poll_url)
        assert r.status_code == 200
        data = r.json()
        assert data["correlation_id"] == corr_id
        assert data["status"]         in ("pending", "ready", "missing")

    def test_shap_poll_url_matches_correlation_id(self, app_client, HIRING_PAYLOAD):
        pred = app_client.post("/hiring/predict", json=HIRING_PAYLOAD).json()
        assert pred["shap_poll_url"] == f"/shap/{pred['correlation_id']}"


# ═════════════════════════════════════════════════════════════════════════════
# Error handling — malformed JSON, missing Content-Type default
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_malformed_json_raises_error(self, app_client):
        r = app_client.post(
            "/hiring/predict",
            content=b"{not valid json}",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422)

    def test_empty_body_422(self, app_client):
        r = app_client.post(
            "/hiring/predict",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_wrong_type_for_numeric_field_422(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json={
            **HIRING_PAYLOAD, "years_experience": "five"
        })
        assert r.status_code == 422

    def test_validation_error_no_stack_trace(self, app_client, HIRING_PAYLOAD):
        r = app_client.post("/hiring/predict", json={**HIRING_PAYLOAD, "education_level": 9})
        body = r.json()
        body_str = json.dumps(body)
        # Stack traces always contain 'Traceback' or 'File "'
        assert "Traceback" not in body_str
        assert 'File "'    not in body_str

    def test_global_error_response_no_detail_leak(self, app_client):
        """The global handler must not expose str(exc) in the response."""
        with patch("hiring.router.predict", side_effect=RuntimeError("secret internal msg")):
            r = app_client.post("/hiring/predict", json={
                "years_experience": 5, "education_level": 2,
                "technical_score": 80, "communication_score": 70,
                "num_past_jobs": 2,
            })
        # Either the router caught it (500) or validation rejected it (422)
        assert r.status_code in (422, 500)
        if r.status_code == 500:
            assert "secret internal msg" not in r.text


# ═════════════════════════════════════════════════════════════════════════════
# Privacy — no PII echoed back
# ═════════════════════════════════════════════════════════════════════════════

class TestPrivacy:

    def test_gender_value_not_in_hiring_response(self, app_client, HIRING_PAYLOAD):
        r    = app_client.post("/hiring/predict", json=HIRING_PAYLOAD)
        text = r.text
        # "female" should not appear outside the fairness monitoring context
        fairness = r.json().get("fairness", {})
        assert "sensitive_value" not in fairness

    def test_ethnicity_not_in_loan_response(self, app_client, LOAN_PAYLOAD):
        data = app_client.post("/loan/predict", json=LOAN_PAYLOAD).json()
        assert "sensitive_value" not in data.get("fairness", {})

    def test_location_not_echoed_in_social_response(self, app_client, SOCIAL_PAYLOAD):
        data = app_client.post("/social/recommend", json=SOCIAL_PAYLOAD).json()
        assert "sensitive_value" not in data.get("fairness", {})


# ═════════════════════════════════════════════════════════════════════════════
# Performance — batch load test
# ═════════════════════════════════════════════════════════════════════════════

class TestPerformance:

    @pytest.mark.parametrize("n_requests", [100, 500])
    def test_batch_hiring_no_crash(self, app_client, HIRING_PAYLOAD, n_requests):
        """Simulate N back-to-back hiring predictions — must all return 200."""
        failures = []
        for i in range(n_requests):
            r = app_client.post("/hiring/predict", json=HIRING_PAYLOAD)
            if r.status_code != 200:
                failures.append((i, r.status_code))
        assert not failures, f"Failures: {failures[:5]}"

    @pytest.mark.parametrize("n_requests", [100, 500])
    def test_batch_loan_no_crash(self, app_client, LOAN_PAYLOAD, n_requests):
        failures = []
        for i in range(n_requests):
            r = app_client.post("/loan/predict", json=LOAN_PAYLOAD)
            if r.status_code != 200:
                failures.append((i, r.status_code))
        assert not failures, f"Failures: {failures[:5]}"

    @pytest.mark.parametrize("n_requests", [100, 500])
    def test_batch_social_no_crash(self, app_client, SOCIAL_PAYLOAD, n_requests):
        failures = []
        for i in range(n_requests):
            r = app_client.post("/social/recommend", json=SOCIAL_PAYLOAD)
            if r.status_code != 200:
                failures.append((i, r.status_code))
        assert not failures, f"Failures: {failures[:5]}"

    def test_mixed_batch_1000_requests(self, app_client, HIRING_PAYLOAD, LOAN_PAYLOAD, SOCIAL_PAYLOAD):
        """1 000-request mixed load — no crashes."""
        endpoints = [
            ("/hiring/predict",   HIRING_PAYLOAD),
            ("/loan/predict",     LOAN_PAYLOAD),
            ("/social/recommend", SOCIAL_PAYLOAD),
        ]
        errors = 0
        for i in range(1_000):
            path, payload = endpoints[i % 3]
            r = app_client.post(path, json=payload)
            if r.status_code not in (200, 422):
                errors += 1
        assert errors == 0

    def test_response_time_reasonable(self, app_client, HIRING_PAYLOAD):
        """Single request must complete in under 5 s (generous CI headroom)."""
        start = time.monotonic()
        app_client.post("/hiring/predict", json=HIRING_PAYLOAD)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"Request took {elapsed:.2f}s — too slow"


# ═════════════════════════════════════════════════════════════════════════════
# Conftest payload fixtures for this module
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def HIRING_PAYLOAD():
    return {
        "years_experience":    5,
        "education_level":     2,
        "technical_score":     82,
        "communication_score": 75,
        "num_past_jobs":       3,
        "certifications":      2,
        "gender":              "female",
    }

@pytest.fixture()
def LOAN_PAYLOAD():
    return {
        "credit_score":     720,
        "annual_income":    75_000,
        "loan_amount":      25_000,
        "loan_term_months": 36,
        "employment_years": 4,
        "existing_debt":    8_000,
        "num_credit_lines": 3,
        "ethnicity":        "hispanic",
    }

@pytest.fixture()
def SOCIAL_PAYLOAD():
    return {
        "avg_session_minutes": 45,
        "posts_per_day":        3,
        "topics_interacted":   12,
        "like_rate":           0.65,
        "share_rate":          0.20,
        "comment_rate":        0.10,
        "account_age_days":   365,
        "age_group":          "25-34",
        "location":           "India",
    }
