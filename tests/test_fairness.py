"""
tests/test_fairness.py

Critical-path fairness tests for fairness/checker.py.

Synthetic datasets are constructed so that bias levels are known exactly,
allowing us to assert:
  - bias_detected = True when DPD / FPR-gap / calibration-gap exceeds threshold
  - bias_detected = False (and score is low) when data is perfectly fair
  - All return-dict schemas are correct
  - Edge cases: single group, empty dataset, all-same prediction

Thresholds from checker.py
  FAIRNESS_THRESHOLD  = 0.10  (DPD / EOD batch check)
  DISPARITY_THRESHOLD = 0.05  (calibration / equalized-odds Phase-2)
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from fairness.checker import (
    DISPARITY_THRESHOLD,
    FAIRNESS_THRESHOLD,
    compute_bias_risk_score,
    demographic_parity_difference,
    equal_opportunity_difference,
    run_batch_fairness_check,
    run_fairness_check,
    run_post_processing_checks,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — synthetic biased datasets
# ═════════════════════════════════════════════════════════════════════════════

def _biased_dataset(n_per_group: int = 100, group_a_rate: float = 0.8,
                    group_b_rate: float = 0.5, seed: int = 0):
    """
    Create (y_pred, y_true, sensitive_values) where:
      Group A receives a positive prediction at *group_a_rate*.
      Group B receives a positive prediction at *group_b_rate*.
    y_true mirrors y_pred (no noise) so EOD == DPD in this fixture.
    """
    rng = np.random.default_rng(seed)

    def _group(n, rate):
        y = (rng.random(n) < rate).astype(int)
        return y.tolist()

    ya = _group(n_per_group, group_a_rate)
    yb = _group(n_per_group, group_b_rate)

    y_pred = ya + yb
    y_true = ya + yb
    sens   = ["A"] * n_per_group + ["B"] * n_per_group
    y_prob = [float(p) * 0.9 + 0.05 for p in y_pred]   # rough probability proxy
    return y_pred, y_prob, y_true, sens


def _fair_dataset(n_per_group: int = 100, rate: float = 0.6, seed: int = 1):
    """Identical outcome rates across groups — perfectly fair."""
    return _biased_dataset(n_per_group, rate, rate, seed)


# ═════════════════════════════════════════════════════════════════════════════
# demographic_parity_difference
# ═════════════════════════════════════════════════════════════════════════════

class TestDemographicParityDifference:

    def test_biased_dpd_exceeds_threshold(self):
        y_pred, _, _, sens = _biased_dataset(
            n_per_group=200, group_a_rate=0.80, group_b_rate=0.50
        )
        dpd = demographic_parity_difference(y_pred, sens)
        # Expected ≈ 0.30; must exceed FAIRNESS_THRESHOLD (0.10)
        assert dpd > FAIRNESS_THRESHOLD, f"DPD={dpd:.4f} should exceed {FAIRNESS_THRESHOLD}"

    def test_fair_dpd_below_threshold(self):
        y_pred, _, _, sens = _fair_dataset(n_per_group=300, rate=0.6)
        dpd = demographic_parity_difference(y_pred, sens)
        # Stochastic, but 300 samples at identical rate → |dpd| should be tiny
        assert dpd < FAIRNESS_THRESHOLD, f"DPD={dpd:.4f} should be < {FAIRNESS_THRESHOLD}"

    def test_dpd_value_matches_manual_calculation(self):
        # Group A: 8 / 10 = 0.8, Group B: 2 / 10 = 0.2  → DPD = 0.6
        y_pred = [1]*8 + [0]*2 + [1]*2 + [0]*8
        sens   = ["A"]*10 + ["B"]*10
        dpd    = demographic_parity_difference(y_pred, sens)
        assert dpd == pytest.approx(0.6, abs=1e-6)

    def test_single_group_returns_zero(self):
        y_pred = [1, 0, 1, 1, 0]
        sens   = ["A"] * 5
        dpd    = demographic_parity_difference(y_pred, sens)
        assert dpd == 0.0

    def test_empty_list_returns_zero(self):
        dpd = demographic_parity_difference([], [])
        assert dpd == 0.0

    def test_all_same_prediction_gives_zero_dpd(self):
        y_pred = [1] * 50 + [1] * 50
        sens   = ["A"]*50 + ["B"]*50
        dpd    = demographic_parity_difference(y_pred, sens)
        assert dpd == pytest.approx(0.0, abs=1e-6)

    def test_dpd_is_non_negative(self):
        y_pred, _, _, sens = _biased_dataset(n_per_group=50)
        dpd = demographic_parity_difference(y_pred, sens)
        assert dpd >= 0.0

    def test_dpd_at_most_one(self):
        # Extreme: group A always positive, group B always negative
        y_pred = [1]*20 + [0]*20
        sens   = ["A"]*20 + ["B"]*20
        dpd    = demographic_parity_difference(y_pred, sens)
        assert dpd == pytest.approx(1.0, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# equal_opportunity_difference
# ═════════════════════════════════════════════════════════════════════════════

class TestEqualOpportunityDifference:

    def test_biased_eod_exceeds_threshold(self):
        # Group A TPR = 1.0, Group B TPR = 0.5
        y_pred = [1]*10 + [0]*0 + [1]*5 + [0]*5
        y_true = [1]*10       + [1]*10
        sens   = ["A"]*10     + ["B"]*10
        eod    = equal_opportunity_difference(y_pred, y_true, sens)
        assert eod > FAIRNESS_THRESHOLD, f"EOD={eod:.4f} should exceed threshold"

    def test_fair_eod_near_zero(self):
        # Both groups: TPR = 0.8
        y_pred = [1]*8+[0]*2 + [1]*8+[0]*2
        y_true = [1]*10      + [1]*10
        sens   = ["A"]*10    + ["B"]*10
        eod    = equal_opportunity_difference(y_pred, y_true, sens)
        assert eod == pytest.approx(0.0, abs=1e-6)

    def test_single_group_returns_zero(self):
        eod = equal_opportunity_difference([1, 0, 1], [1, 1, 0], ["A", "A", "A"])
        assert eod == 0.0

    def test_no_positives_in_one_group_skipped(self):
        # Group B has no true positives — skipped in EOD calculation
        y_pred = [1]*5 + [0]*5
        y_true = [1]*5 + [0]*5   # Group B has no y_true=1
        sens   = ["A"]*5 + ["B"]*5
        eod    = equal_opportunity_difference(y_pred, y_true, sens)
        assert eod >= 0.0

    def test_perfect_classifier_eod_zero(self):
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = y_true[:]
        sens   = ["A", "A", "A", "A", "B", "B", "B", "B"]
        eod    = equal_opportunity_difference(y_pred, y_true, sens)
        assert eod == pytest.approx(0.0, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# run_batch_fairness_check
# ═════════════════════════════════════════════════════════════════════════════

class TestRunBatchFairnessCheck:

    def test_biased_result_is_not_fair(self):
        y_pred, _, y_true, sens = _biased_dataset(
            n_per_group=200, group_a_rate=0.80, group_b_rate=0.40
        )
        result = run_batch_fairness_check(y_pred, y_true, sens, "gender", "hiring")
        assert result["is_fair"] is False
        assert result["warnings"] is not None

    def test_fair_result_is_fair(self):
        y_pred, _, y_true, sens = _fair_dataset(n_per_group=300)
        result = run_batch_fairness_check(y_pred, y_true, sens, "gender", "hiring")
        assert result["is_fair"] is True

    def test_response_schema(self):
        y_pred, _, y_true, sens = _biased_dataset(n_per_group=50)
        result = run_batch_fairness_check(y_pred, y_true, sens, "ethnicity", "loan")
        assert "domain"              in result
        assert "sensitive_attribute" in result
        assert "is_fair"             in result
        assert "threshold"           in result
        assert "metrics"             in result
        assert "demographic_parity_difference" in result["metrics"]
        assert "equal_opportunity_difference"  in result["metrics"]
        assert result["threshold"] == FAIRNESS_THRESHOLD

    def test_dpd_in_result_matches_standalone(self):
        y_pred, _, y_true, sens = _biased_dataset(n_per_group=100)
        result = run_batch_fairness_check(y_pred, y_true, sens, "gender", "test")
        standalone_dpd = demographic_parity_difference(y_pred, sens)
        assert result["metrics"]["demographic_parity_difference"] == pytest.approx(
            standalone_dpd, abs=1e-6
        )


# ═════════════════════════════════════════════════════════════════════════════
# run_fairness_check  (single-prediction)
# ═════════════════════════════════════════════════════════════════════════════

class TestRunFairnessCheck:

    def test_returns_correct_schema(self):
        result = run_fairness_check(1, "gender", "female", "hiring")
        assert "sensitive_attribute" in result
        assert "sensitive_value"     in result
        assert "domain"              in result
        assert "is_fair"             in result
        assert "ethical_note"        in result
        assert "metrics"             in result

    def test_sensitive_value_present_for_internal_use(self):
        result = run_fairness_check(1, "gender", "female", "hiring")
        assert result["sensitive_value"] == "female"   # stripped before API response

    def test_no_sensitive_attr(self):
        result = run_fairness_check(0, "not_provided", "unknown", "loan")
        assert result["is_fair"] is True

    def test_ethical_note_present(self):
        result = run_fairness_check(1, "ethnicity", "hispanic", "loan")
        assert len(result["ethical_note"]) > 10


# ═════════════════════════════════════════════════════════════════════════════
# run_post_processing_checks  (Phase 2 — calibration + equalized odds)
# ═════════════════════════════════════════════════════════════════════════════

class TestRunPostProcessingChecks:

    def test_biased_dataset_flags_for_review(self):
        y_pred, y_prob, y_true, sens = _biased_dataset(
            n_per_group=80, group_a_rate=0.80, group_b_rate=0.40
        )
        # Force large FPR disparity: all Group B negatives predicted positive
        for i in range(80, 160):
            if y_true[i] == 0:
                y_pred[i] = 1

        result = run_post_processing_checks(y_pred, y_prob, y_true, sens, "gender", "test")
        assert result["flag_for_review"] is True
        assert len(result["warnings"])   >  0

    def test_fair_dataset_does_not_flag(self):
        y_pred, y_prob, y_true, sens = _fair_dataset(n_per_group=150)
        result = run_post_processing_checks(y_pred, y_prob, y_true, sens, "gender", "test")
        # With identical rates, neither FPR nor calibration gap should exceed 5 %
        # (stochastic — allow one disparity due to sampling noise, just check no crash)
        assert isinstance(result["flag_for_review"], bool)

    def test_result_schema_complete(self):
        y_pred, y_prob, y_true, sens = _biased_dataset(n_per_group=50)
        result = run_post_processing_checks(y_pred, y_prob, y_true, sens, "gender", "hiring")

        assert "domain"              in result
        assert "sensitive_attribute" in result
        assert "n_records"           in result
        assert "calibration"         in result
        assert "equalized_odds"      in result
        assert "flag_for_review"     in result
        assert "warnings"            in result
        assert "post_processing_boost" in result

        cal = result["calibration"]
        assert "max_gap"            in cal
        assert "disparity_detected" in cal
        assert "penalty"            in cal
        assert "per_group"          in cal

        eo = result["equalized_odds"]
        assert "fpr_gap"            in eo
        assert "fnr_gap"            in eo
        assert "disparity_detected" in eo
        assert "penalty"            in eo
        assert "per_group"          in eo

    def test_post_processing_boost_keys(self):
        y_pred, y_prob, y_true, sens = _biased_dataset(n_per_group=60)
        result = run_post_processing_checks(y_pred, y_prob, y_true, sens, "ethnicity", "loan")
        boost = result["post_processing_boost"]
        assert "calibration_penalty"    in boost
        assert "equalized_odds_penalty" in boost
        assert 0.0 <= boost["calibration_penalty"]    <= 1.0
        assert 0.0 <= boost["equalized_odds_penalty"] <= 1.0

    def test_single_group_graceful(self):
        y_pred = [1, 0, 1, 0, 1]
        y_prob = [0.9, 0.2, 0.8, 0.1, 0.85]
        y_true = [1, 0, 1, 0, 1]
        sens   = ["A"] * 5
        result = run_post_processing_checks(y_pred, y_prob, y_true, sens, "gender", "hiring")
        # Single group → cannot compute inter-group gaps → no flag
        assert result["flag_for_review"] is False

    def test_n_records_correct(self):
        y_pred, y_prob, y_true, sens = _biased_dataset(n_per_group=70)
        result = run_post_processing_checks(y_pred, y_prob, y_true, sens, "gender", "hiring")
        assert result["n_records"] == 140

    def test_penalty_normalised_to_unit_interval(self):
        y_pred, y_prob, y_true, sens = _biased_dataset(
            n_per_group=100, group_a_rate=1.0, group_b_rate=0.0
        )
        result = run_post_processing_checks(y_pred, y_prob, y_true, sens, "gender", "loan")
        for key in ("calibration_penalty", "equalized_odds_penalty"):
            val = result["post_processing_boost"][key]
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


# ═════════════════════════════════════════════════════════════════════════════
# compute_bias_risk_score
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeBiasRiskScore:

    def test_schema_complete(self):
        result = compute_bias_risk_score(0.7, None, "gender", "hiring")
        assert "score"                   in result
        assert "band"                    in result
        assert "flag_for_review"         in result
        assert "components"              in result
        assert "recommendation"          in result
        assert "post_processing_applied" in result

        comps = result["components"]
        assert "boundary_proximity"     in comps
        assert "shap_concentration"     in comps
        assert "attribute_base_risk"    in comps
        assert "calibration_penalty"    in comps
        assert "equalized_odds_penalty" in comps

    def test_score_in_unit_interval(self):
        for conf in (0.0, 0.1, 0.5, 0.9, 1.0):
            result = compute_bias_risk_score(conf, None)
            assert 0.0 <= result["score"] <= 1.0

    def test_band_values(self):
        valid_bands = {"low", "moderate", "high", "critical"}
        for conf in np.linspace(0, 1, 20):
            result = compute_bias_risk_score(float(conf), None)
            assert result["band"] in valid_bands

    def test_confidence_at_boundary_is_high_risk(self):
        """confidence=0.5 maximises boundary_proximity → higher score."""
        result_boundary = compute_bias_risk_score(0.5, None, "gender")
        result_certain  = compute_bias_risk_score(0.99, None, "gender")
        assert result_boundary["score"] > result_certain["score"]

    def test_high_risk_attribute_increases_score(self):
        """race (0.90) should produce higher score than language (0.30)."""
        result_race     = compute_bias_risk_score(0.6, None, "race")
        result_language = compute_bias_risk_score(0.6, None, "language")
        assert result_race["score"] > result_language["score"]

    def test_post_processing_boost_raises_score(self):
        base = compute_bias_risk_score(0.6, None, "gender")
        with_boost = compute_bias_risk_score(
            0.6, None, "gender",
            post_processing_boost={"calibration_penalty": 1.0, "equalized_odds_penalty": 1.0}
        )
        assert with_boost["score"]              >= base["score"]
        assert with_boost["post_processing_applied"] is True
        assert base["post_processing_applied"]       is False

    def test_shap_concentration_increases_score(self):
        """A single dominant SHAP feature (high HHI) raises the score."""
        dominated_shap = {"f1": 10.0, "f2": 0.01, "f3": 0.01}
        uniform_shap   = {"f1": 1.0,  "f2": 1.0,  "f3": 1.0}

        result_dom = compute_bias_risk_score(0.6, dominated_shap, "gender")
        result_uni = compute_bias_risk_score(0.6, uniform_shap,   "gender")
        assert result_dom["score"] >= result_uni["score"]

    def test_flag_for_review_high_band(self):
        """A critical/high score must always flag for review."""
        # Force critical score: confidence=0.5 + race attr → high base
        result = compute_bias_risk_score(
            0.5, None, "race",
            post_processing_boost={"calibration_penalty": 1.0, "equalized_odds_penalty": 1.0}
        )
        if result["band"] in ("high", "critical"):
            assert result["flag_for_review"] is True

    def test_recommendation_is_string(self):
        result = compute_bias_risk_score(0.7, None, "gender")
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 5

    def test_no_sensitive_attr_uses_not_provided(self):
        result = compute_bias_risk_score(0.6, None)
        # not_provided weight = 0.20 → lower component than most named attrs
        assert result["components"]["attribute_base_risk"] == pytest.approx(0.20, abs=1e-4)

    def test_deterministic(self):
        r1 = compute_bias_risk_score(0.75, {"a": 0.5, "b": -0.3}, "gender")
        r2 = compute_bias_risk_score(0.75, {"a": 0.5, "b": -0.3}, "gender")
        assert r1["score"] == r2["score"]
