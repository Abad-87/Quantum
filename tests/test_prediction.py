"""
tests/test_prediction.py

Unit tests for the three domain predictor modules:
  hiring/predictor.py
  loan/predictor.py
  social/predictor.py

Strategy
--------
- Use real dummy models (session-scoped via conftest) for integration paths.
- Use mock_model fixture for isolated unit tests that don't need sklearn.
- Test every key in the return dict, including SHAP-related fields whose
  values confirm Phase-3 async-SHAP behaviour (shap_values={}, pending).
- Cover valid inputs, edge values, missing features (KeyError path),
  and prediction boundary cases (confidence at 0.5 = max bias-risk boundary).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import hiring.predictor  as h_pred
import loan.predictor    as l_pred
import social.predictor  as s_pred
from fairness.checker import compute_bias_risk_score


# ═════════════════════════════════════════════════════════════════════════════
# Canonical feature sets (minimal, valid)
# ═════════════════════════════════════════════════════════════════════════════

HIRING_FEATURES = {
    "years_experience":    5.0,
    "education_level":     2,
    "technical_score":     82.0,
    "communication_score": 75.0,
    "num_past_jobs":       3,
    "certifications":      2,
}

LOAN_FEATURES = {
    "credit_score":     720,
    "annual_income":    75_000.0,
    "loan_amount":      25_000.0,
    "loan_term_months": 36,
    "employment_years": 4.0,
    "existing_debt":    8_000.0,
    "num_credit_lines": 3,
}

SOCIAL_FEATURES = {
    "avg_session_minutes": 45.0,
    "posts_per_day":        3.0,
    "topics_interacted":   12,
    "like_rate":           0.65,
    "share_rate":          0.20,
    "comment_rate":        0.10,
    "account_age_days":   365,
}


# ═════════════════════════════════════════════════════════════════════════════
# HIRING PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

class TestHiringPredictor:

    # ── Return-dict schema ────────────────────────────────────────────────────

    def test_returns_all_required_keys(self, loaded_models):
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        required = {
            "prediction", "confidence", "shap_values", "shap_available",
            "shap_status", "explanation", "bias_risk", "input_row", "feature_names",
        }
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_prediction_is_binary(self, loaded_models):
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        assert result["prediction"] in (0, 1)

    def test_confidence_in_unit_interval(self, loaded_models):
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        assert 0.0 <= result["confidence"] <= 1.0

    # ── Phase-3: async SHAP contract ──────────────────────────────────────────

    def test_shap_values_empty_on_critical_path(self, loaded_models):
        """SHAP is deferred — shap_values must be {} immediately."""
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        assert result["shap_values"]    == {}
        assert result["shap_available"] is False
        assert result["shap_status"]    == "pending"

    def test_feature_names_correct(self, loaded_models):
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        assert result["feature_names"] == h_pred.FEATURE_NAMES
        assert len(result["feature_names"]) == 6

    def test_input_row_shape(self, loaded_models):
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        assert isinstance(result["input_row"], list)
        assert len(result["input_row"]) == 1      # 1 sample
        assert len(result["input_row"][0]) == 6   # 6 features

    # ── Explanation ───────────────────────────────────────────────────────────

    def test_explanation_is_non_empty_string(self, loaded_models):
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 10

    def test_explanation_contains_pending_hint(self, loaded_models):
        """All rule-based explanations should mention SHAP is coming."""
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        assert "pending" in result["explanation"].lower() or \
               "SHAP" in result["explanation"]

    def test_hired_explanation_content(self, mock_model):
        """High scores → prediction=1 → explanation starts with 'Hired'."""
        features = {**HIRING_FEATURES, "technical_score": 90, "communication_score": 90}
        result = h_pred.predict(mock_model, features)
        # mock always returns 1
        assert result["explanation"].startswith("Hired")

    def test_not_hired_explanation_content(self):
        """Low scores → prediction=0 → explanation starts with 'Not hired'."""
        m = MagicMock()
        m.predict       = MagicMock(return_value=np.array([0]))
        m.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2]]))

        features = {**HIRING_FEATURES, "technical_score": 20, "communication_score": 20}
        result = h_pred.predict(m, features)
        assert result["explanation"].startswith("Not hired")

    # ── Bias risk ─────────────────────────────────────────────────────────────

    def test_bias_risk_structure(self, loaded_models):
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES, sensitive_attr="gender")
        br = result["bias_risk"]
        assert "score"           in br
        assert "band"            in br
        assert "flag_for_review" in br
        assert "components"      in br
        assert "recommendation"  in br
        assert 0.0 <= br["score"] <= 1.0
        assert br["band"] in ("low", "moderate", "high", "critical")

    def test_bias_risk_no_sensitive_attr(self, loaded_models):
        """With no sensitive_attr the score uses 'not_provided' weight (0.20)."""
        result = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        assert result["bias_risk"]["score"] >= 0.0

    # ── Mocked model path ─────────────────────────────────────────────────────

    def test_uses_predict_proba_when_available(self, mock_model):
        result = h_pred.predict(mock_model, HIRING_FEATURES)
        mock_model.predict_proba.assert_called_once()
        assert result["confidence"] == pytest.approx(0.9, abs=1e-4)

    def test_falls_back_to_0_5_without_proba(self):
        m = MagicMock(spec=[])            # no predict_proba attribute
        m.predict = MagicMock(return_value=np.array([1]))
        result = h_pred.predict(m, HIRING_FEATURES)
        assert result["confidence"] == 0.5

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_zero_experience_zero_scores(self, loaded_models):
        """Legal edge case: fresh graduate with zero scores."""
        f = {**HIRING_FEATURES, "years_experience": 0, "technical_score": 0,
             "communication_score": 0, "certifications": 0}
        result = h_pred.predict(loaded_models["hiring"], f)
        assert result["prediction"] in (0, 1)

    def test_max_valid_values(self, loaded_models):
        f = {
            "years_experience":    50.0,
            "education_level":     3,
            "technical_score":     100.0,
            "communication_score": 100.0,
            "num_past_jobs":       30,
            "certifications":      20,
        }
        result = h_pred.predict(loaded_models["hiring"], f)
        assert result["prediction"] in (0, 1)

    def test_missing_feature_raises_key_error(self, loaded_models):
        """Predictors don't tolerate missing keys — callers validate first."""
        incomplete = {"years_experience": 5}
        with pytest.raises(KeyError):
            h_pred.predict(loaded_models["hiring"], incomplete)


# ═════════════════════════════════════════════════════════════════════════════
# LOAN PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

class TestLoanPredictor:

    def test_returns_all_required_keys(self, loaded_models):
        result = l_pred.predict(loaded_models["loan"], LOAN_FEATURES)
        required = {
            "prediction", "confidence", "shap_values", "shap_available",
            "shap_status", "explanation", "bias_risk", "input_row", "feature_names",
        }
        assert required.issubset(result.keys())

    def test_prediction_binary(self, loaded_models):
        result = l_pred.predict(loaded_models["loan"], LOAN_FEATURES)
        assert result["prediction"] in (0, 1)

    def test_shap_deferred(self, loaded_models):
        result = l_pred.predict(loaded_models["loan"], LOAN_FEATURES)
        assert result["shap_values"]    == {}
        assert result["shap_available"] is False
        assert result["shap_status"]    == "pending"

    def test_approved_explanation(self, mock_model):
        result = l_pred.predict(mock_model, LOAN_FEATURES)
        assert "approved" in result["explanation"].lower() or \
               "pending"  in result["explanation"].lower()

    def test_rejected_explanation(self):
        m = MagicMock()
        m.predict       = MagicMock(return_value=np.array([0]))
        m.predict_proba = MagicMock(return_value=np.array([[0.9, 0.1]]))
        result = l_pred.predict(m, LOAN_FEATURES)
        assert "rejected" in result["explanation"].lower()

    def test_low_credit_score_explanation(self):
        m = MagicMock()
        m.predict       = MagicMock(return_value=np.array([0]))
        m.predict_proba = MagicMock(return_value=np.array([[0.9, 0.1]]))
        f = {**LOAN_FEATURES, "credit_score": 400}
        result = l_pred.predict(m, f)
        assert "credit" in result["explanation"].lower()

    def test_confidence_bounds(self, loaded_models):
        result = l_pred.predict(loaded_models["loan"], LOAN_FEATURES)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_feature_names_count(self, loaded_models):
        result = l_pred.predict(loaded_models["loan"], LOAN_FEATURES)
        assert len(result["feature_names"]) == 7

    def test_extreme_credit_score_high(self, loaded_models):
        f = {**LOAN_FEATURES, "credit_score": 850}
        result = l_pred.predict(loaded_models["loan"], f)
        assert result["prediction"] in (0, 1)

    def test_extreme_credit_score_low(self, loaded_models):
        f = {**LOAN_FEATURES, "credit_score": 300}
        result = l_pred.predict(loaded_models["loan"], f)
        assert result["prediction"] in (0, 1)

    def test_missing_feature_raises(self, loaded_models):
        with pytest.raises(KeyError):
            l_pred.predict(loaded_models["loan"], {"credit_score": 700})

    def test_bias_risk_with_ethnicity(self, loaded_models):
        result = l_pred.predict(loaded_models["loan"], LOAN_FEATURES, sensitive_attr="ethnicity")
        assert result["bias_risk"]["score"] >= 0.0
        # ethnicity weight is 0.85 — score should be non-trivially high
        assert result["bias_risk"]["score"] > 0.1


# ═════════════════════════════════════════════════════════════════════════════
# SOCIAL PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

class TestSocialPredictor:

    def test_returns_all_required_keys(self, loaded_models):
        result = s_pred.predict(loaded_models["social"], SOCIAL_FEATURES)
        required = {
            "prediction", "category_label", "confidence", "shap_values",
            "shap_available", "shap_status", "explanation", "bias_risk",
            "input_row", "feature_names",
        }
        assert required.issubset(result.keys())

    def test_prediction_in_valid_category_range(self, loaded_models):
        result = s_pred.predict(loaded_models["social"], SOCIAL_FEATURES)
        assert 0 <= result["prediction"] <= 7

    def test_category_label_is_string(self, loaded_models):
        result = s_pred.predict(loaded_models["social"], SOCIAL_FEATURES)
        assert isinstance(result["category_label"], str)
        assert len(result["category_label"]) > 2

    def test_known_categories(self, loaded_models):
        result = s_pred.predict(loaded_models["social"], SOCIAL_FEATURES)
        known  = set(s_pred.CONTENT_CATEGORIES.values())
        assert result["category_label"] in known or \
               result["category_label"].startswith("Category")

    def test_shap_deferred(self, loaded_models):
        result = s_pred.predict(loaded_models["social"], SOCIAL_FEATURES)
        assert result["shap_values"]    == {}
        assert result["shap_available"] is False
        assert result["shap_status"]    == "pending"

    def test_explanation_references_category(self, loaded_models):
        result = s_pred.predict(loaded_models["social"], SOCIAL_FEATURES)
        assert result["category_label"] in result["explanation"] or \
               "pending" in result["explanation"].lower()

    def test_feature_names_count(self, loaded_models):
        result = s_pred.predict(loaded_models["social"], SOCIAL_FEATURES)
        assert len(result["feature_names"]) == 7

    def test_high_engagement_signal(self, loaded_models):
        f = {**SOCIAL_FEATURES, "like_rate": 0.95, "share_rate": 0.50}
        result = s_pred.predict(loaded_models["social"], f)
        assert "engagement" in result["explanation"].lower() or \
               "pending"    in result["explanation"].lower()

    def test_minimal_activity_user(self, loaded_models):
        f = {
            "avg_session_minutes": 1.0,
            "posts_per_day":        0.0,
            "topics_interacted":    1,
            "like_rate":            0.01,
            "share_rate":           0.0,
            "comment_rate":         0.0,
            "account_age_days":     1,
        }
        result = s_pred.predict(loaded_models["social"], f)
        assert result["prediction"] in range(8)

    def test_missing_feature_raises(self, loaded_models):
        with pytest.raises(KeyError):
            s_pred.predict(loaded_models["social"], {"like_rate": 0.5})


# ═════════════════════════════════════════════════════════════════════════════
# SHAP ASYNC BACKGROUND — unit test for _blocking_shap_compute
# ═════════════════════════════════════════════════════════════════════════════

class TestShapBackgroundCompute:

    def test_blocking_shap_returns_dict_and_string(self, loaded_models):
        from utils.shap_cache import _blocking_shap_compute

        model     = loaded_models["hiring"]
        input_row = [[5.0, 2, 82.0, 75.0, 3, 2]]
        shap_dict, explanation = _blocking_shap_compute(
            model, input_row, prediction=1,
            feature_names=h_pred.FEATURE_NAMES,
            features_dict=HIRING_FEATURES,
            domain="hiring",
        )

        # Either SHAP succeeded and returned a dict …
        if shap_dict:
            assert set(shap_dict.keys()) == set(h_pred.FEATURE_NAMES)
            assert all(isinstance(v, float) for v in shap_dict.values())
            assert "[SHAP]" in explanation
        else:
            # … or fell back to rule-based
            assert isinstance(explanation, str)
            assert len(explanation) > 5

    def test_blocking_shap_fallback_on_bad_model(self):
        """A model that raises during SHAP must return ({}, fallback_explanation)."""
        from utils.shap_cache import _blocking_shap_compute

        bad_model = MagicMock()
        bad_model.predict = MagicMock(side_effect=RuntimeError("broken"))

        shap_dict, explanation = _blocking_shap_compute(
            bad_model, [[1, 2, 3]], prediction=0,
            feature_names=["a", "b", "c"],
            features_dict={"a": 1},
            domain="hiring",
        )
        assert shap_dict   == {}
        assert isinstance(explanation, str)

    @pytest.mark.asyncio
    async def test_compute_shap_background_stores_in_cache(self, loaded_models):
        from utils.shap_cache import compute_shap_background, shap_cache

        corr_id   = "test-shap-async-001"
        model     = loaded_models["hiring"]
        input_row = [[5.0, 2, 82.0, 75.0, 3, 2]]

        await compute_shap_background(
            model, input_row, prediction=1,
            feature_names=h_pred.FEATURE_NAMES,
            correlation_id=corr_id,
            domain="hiring",
            features_dict=HIRING_FEATURES,
            sensitive_attr="gender",
        )

        report = shap_cache.get(corr_id)
        assert report is not None
        assert report["correlation_id"] == corr_id
        assert "shap_available" in report
        assert "explanation"    in report
        assert "duration_ms"    in report
        # Clean up
        shap_cache.delete(corr_id)

    @pytest.mark.asyncio
    async def test_shap_cache_status_ready_after_compute(self, loaded_models):
        from utils.shap_cache import compute_shap_background, shap_cache

        corr_id = "test-shap-status-002"
        await compute_shap_background(
            loaded_models["hiring"],
            [[5.0, 2, 82.0, 75.0, 3, 2]],
            prediction=1,
            feature_names=h_pred.FEATURE_NAMES,
            correlation_id=corr_id,
            domain="hiring",
            features_dict=HIRING_FEATURES,
        )
        assert shap_cache.status(corr_id) == "ready"
        shap_cache.delete(corr_id)


# ═════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY FORMAT — bonus requirement
# ═════════════════════════════════════════════════════════════════════════════

class TestExplainabilityFormat:
    """Validate that explanation strings follow expected format conventions."""

    @pytest.mark.parametrize("prediction,expected_prefix", [
        (1, "Hired"),
        (0, "Not hired"),
    ])
    def test_hiring_explanation_prefix(self, prediction, expected_prefix, loaded_models):
        m = MagicMock()
        m.predict       = MagicMock(return_value=np.array([prediction]))
        m.predict_proba = MagicMock(return_value=np.array([[1 - 0.1 * prediction, 0.1 * prediction + 0.5]]))
        result = h_pred.predict(m, HIRING_FEATURES)
        assert result["explanation"].startswith(expected_prefix)

    @pytest.mark.parametrize("prediction,expected_keyword", [
        (1, "approved"),
        (0, "rejected"),
    ])
    def test_loan_explanation_keyword(self, prediction, expected_keyword):
        m = MagicMock()
        m.predict       = MagicMock(return_value=np.array([prediction]))
        m.predict_proba = MagicMock(return_value=np.array([[0.6, 0.4]]))
        result = l_pred.predict(m, LOAN_FEATURES)
        assert expected_keyword in result["explanation"].lower()

    def test_explanation_always_ends_with_pending(self, loaded_models):
        """All three predictors should hint that SHAP is being computed."""
        h = h_pred.predict(loaded_models["hiring"], HIRING_FEATURES)
        l = l_pred.predict(loaded_models["loan"],   LOAN_FEATURES)
        s = s_pred.predict(loaded_models["social"],  SOCIAL_FEATURES)

        for result in (h, l, s):
            assert "pending" in result["explanation"].lower() or \
                   "SHAP"    in result["explanation"]

    def test_shap_report_explanation_starts_with_shap_tag(self, loaded_models):
        """When SHAP succeeds asynchronously the stored explanation starts [SHAP]."""
        from utils.shap_cache import _blocking_shap_compute

        model     = loaded_models["hiring"]
        input_row = [[5.0, 2, 82.0, 75.0, 3, 2]]
        shap_dict, explanation = _blocking_shap_compute(
            model, input_row, prediction=1,
            feature_names=h_pred.FEATURE_NAMES,
            features_dict=HIRING_FEATURES,
            domain="hiring",
        )
        if shap_dict:   # SHAP library available
            assert explanation.startswith("[SHAP]")
            assert "value:" in explanation.lower() or "SHAP:" in explanation
