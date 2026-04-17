"""
tests/test_database_logger.py

Tests for utils/database.py, utils/logger.py, and utils/pii.py.

No real MongoDB is used anywhere in this file.  All Motor/pymongo calls are
mocked.  JSON fallback paths are tested against real temporary files.

Coverage targets
----------------
Database
  - save_prediction: MongoDB path, JSON fallback, timestamp injection
  - get_recent_predictions: MongoDB path, JSON fallback, domain filter,
    sensitive_attr filter, lean projection
  - preprocess_features: cold-start passthrough, correlation neutralisation
  - ensure_indexes: MongoDB skip (None db), no crash when called without Mongo

Logger
  - log_correlation_event: audit record written to JSONL file
  - log_prediction: console record emitted without PII
  - PII is masked in every write path

PII Masker (utils/pii.py)
  - Key-name masking (50+ fields)
  - Allowlist never masked (model features, system fields)
  - Value-pattern regex scrubbing (email, phone, card, SSN, IP, date)
  - Deterministic pseudonymisation
  - Deep traversal: nested dicts, lists, tuples, sets
  - PII_MASK_ENABLED=false disables masking
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# PII MASKER
# ═════════════════════════════════════════════════════════════════════════════

class TestPIIMasker:
    """Tests for utils/pii.py PIIMasker."""

    @pytest.fixture(autouse=True)
    def ensure_masking_enabled(self, monkeypatch):
        monkeypatch.setenv("PII_MASK_ENABLED", "true")
        # Re-import to pick up env change
        import importlib
        import utils.pii as pii_mod
        importlib.reload(pii_mod)
        self.masker = pii_mod.PIIMasker()

    # ── Key-name masking ──────────────────────────────────────────────────────

    def test_email_key_masked(self):
        result = self.masker.mask({"email": "alice@example.com"})
        assert result["email"].startswith("<MASKED:")

    def test_password_key_masked(self):
        result = self.masker.mask({"password": "hunter2"})
        assert result["password"].startswith("<MASKED:")

    def test_ssn_key_masked(self):
        result = self.masker.mask({"ssn": "123-45-6789"})
        assert result["ssn"].startswith("<MASKED:")

    def test_sensitive_value_key_masked(self):
        result = self.masker.mask({"sensitive_value": "female"})
        assert result["sensitive_value"].startswith("<MASKED:")

    def test_card_number_masked(self):
        result = self.masker.mask({"card_number": "4111111111111111"})
        assert result["card_number"].startswith("<MASKED:")

    def test_phone_key_masked(self):
        result = self.masker.mask({"phone": "+1 555 867 5309"})
        assert result["phone"].startswith("<MASKED:")

    # ── Allowlist — never masked ──────────────────────────────────────────────

    def test_credit_score_not_masked(self):
        result = self.masker.mask({"credit_score": 720})
        assert result["credit_score"] == 720

    def test_technical_score_not_masked(self):
        result = self.masker.mask({"technical_score": 85.0})
        assert result["technical_score"] == 85.0

    def test_correlation_id_not_masked(self):
        cid = "some-uuid-string"
        result = self.masker.mask({"correlation_id": cid})
        assert result["correlation_id"] == cid

    def test_domain_not_masked(self):
        result = self.masker.mask({"domain": "hiring"})
        assert result["domain"] == "hiring"

    def test_prediction_int_not_masked(self):
        result = self.masker.mask({"prediction": 1})
        assert result["prediction"] == 1

    def test_is_fair_bool_not_masked(self):
        result = self.masker.mask({"is_fair": True})
        assert result["is_fair"] is True

    # ── Value-pattern regex scrubbing ─────────────────────────────────────────

    def test_email_in_string_scrubbed(self):
        s = self.masker.mask_str("Contact us at alice@example.com please")
        assert "alice@example.com"   not in s
        assert "<REDACTED:EMAIL>"    in s

    def test_ip_address_scrubbed(self):
        s = self.masker.mask_str("Request from 192.168.1.42")
        assert "192.168.1.42"        not in s
        assert "<REDACTED:IP_ADDRESS>" in s

    def test_phone_in_string_scrubbed(self):
        s = self.masker.mask_str("Call +1 555 867 5309 for support")
        assert "REDACTED" in s

    def test_ssn_in_string_scrubbed(self):
        s = self.masker.mask_str("SSN is 123-45-6789")
        assert "123-45-6789" not in s
        assert "<REDACTED:SSN>" in s

    def test_clean_string_unchanged(self):
        s = "Hired — strong technical score (82/100)"
        assert self.masker.mask_str(s) == s

    # ── Deep traversal ────────────────────────────────────────────────────────

    def test_nested_dict_pii_masked(self):
        obj = {"outer": {"password": "secret", "domain": "hiring"}}
        result = self.masker.mask(obj)
        assert result["outer"]["password"].startswith("<MASKED:")
        assert result["outer"]["domain"]   == "hiring"

    def test_list_of_strings_scrubbed(self):
        obj = ["alice@example.com", "normal text", "bob@corp.com"]
        result = self.masker.mask(obj)
        assert "<REDACTED:EMAIL>" in result[0]
        assert result[1]         == "normal text"
        assert "<REDACTED:EMAIL>" in result[2]

    def test_tuple_preserved_as_tuple(self):
        obj = ("alice@example.com", 42)
        result = self.masker.mask(obj)
        assert isinstance(result, tuple)
        assert "<REDACTED:EMAIL>" in result[0]
        assert result[1] == 42

    def test_int_float_none_unchanged(self):
        obj = {"a": 1, "b": 3.14, "c": None}
        result = self.masker.mask(obj)
        assert result["a"] is 1
        assert result["b"] == pytest.approx(3.14)
        assert result["c"] is None

    # ── Pseudonymisation ─────────────────────────────────────────────────────

    def test_pseudonymisation_deterministic(self):
        p1 = self.masker.pseudonymise("alice@example.com")
        p2 = self.masker.pseudonymise("alice@example.com")
        assert p1 == p2

    def test_different_values_different_pseudonyms(self):
        p1 = self.masker.pseudonymise("alice@example.com")
        p2 = self.masker.pseudonymise("bob@example.com")
        assert p1 != p2

    def test_pseudonym_format(self):
        p = self.masker.pseudonymise("test")
        assert p.startswith("<MASKED:")
        assert p.endswith(">")
        inner = p[8:-1]   # strip <MASKED: and >
        assert len(inner) == 8
        assert all(c in "0123456789abcdef" for c in inner)

    # ── is_pii_key ────────────────────────────────────────────────────────────

    def test_known_pii_keys(self):
        for key in ("email", "password", "ssn", "dob", "card_number",
                    "phone", "token", "api_key", "sensitive_value"):
            assert self.masker.is_pii_key(key), f"{key} should be PII"

    def test_known_safe_keys(self):
        for key in ("credit_score", "technical_score", "annual_income",
                    "like_rate", "domain", "prediction", "correlation_id",
                    "is_fair", "sensitive_attribute"):
            assert not self.masker.is_pii_key(key), f"{key} should NOT be PII"

    def test_substring_match(self):
        assert self.masker.is_pii_key("old_password")
        assert self.masker.is_pii_key("auth_token")
        assert self.masker.is_pii_key("user_ssn")


# ═════════════════════════════════════════════════════════════════════════════
# LOGGER
# ═════════════════════════════════════════════════════════════════════════════

class TestLogger:

    @pytest.fixture()
    def tmp_audit(self, tmp_path, monkeypatch):
        """Redirect audit log to a temp file for this test."""
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        import importlib
        import utils.logger as log_mod
        importlib.reload(log_mod)
        return tmp_path / "audit.jsonl"

    def test_log_correlation_event_creates_file(self, tmp_audit):
        from utils.logger import log_correlation_event
        log_correlation_event(
            correlation_id = "test-corr-001",
            event          = "request_received",
            path           = "/hiring/predict",
            method         = "POST",
            payload        = {"technical_score": 80},
            model_metadata = {"version": "abc123"},
            result         = None,
        )
        assert tmp_audit.exists()

    def test_log_correlation_event_valid_json(self, tmp_audit):
        from utils.logger import log_correlation_event
        log_correlation_event(
            correlation_id = "test-corr-002",
            event          = "prediction_complete",
            path           = "/loan/predict",
            method         = "POST",
        )
        lines = tmp_audit.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["correlation_id"] == "test-corr-002"
        assert record["event"]          == "prediction_complete"
        assert record["path"]           == "/loan/predict"
        assert "timestamp_utc"          in record

    def test_audit_is_append_only(self, tmp_audit):
        from utils.logger import log_correlation_event

        for i in range(5):
            log_correlation_event(f"corr-{i}", "request_received", f"/p{i}", "POST")

        lines = tmp_audit.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
        # Each must be valid JSON
        for line in lines:
            obj = json.loads(line)
            assert "correlation_id" in obj

    def test_pii_masked_in_payload(self, tmp_audit, monkeypatch):
        """Email in payload must be pseudonymised in the audit file."""
        monkeypatch.setenv("PII_MASK_ENABLED", "true")
        import importlib
        import utils.logger as log_mod
        importlib.reload(log_mod)

        from utils.logger import log_correlation_event
        log_correlation_event(
            correlation_id = "corr-pii-001",
            event          = "request_received",
            path           = "/hiring/predict",
            method         = "POST",
            payload        = {"email": "alice@example.com", "technical_score": 80},
        )
        content = tmp_audit.read_text(encoding="utf-8")
        assert "alice@example.com" not in content
        # pseudonym or REDACTED must appear
        assert "<MASKED:" in content or "<REDACTED:" in content

    def test_log_prediction_does_not_raise(self, tmp_audit):
        from utils.logger import log_prediction
        # Should not raise even with strange input
        log_prediction(
            domain           = "hiring",
            input_data       = {"technical_score": 80, "years_experience": 5},
            prediction       = 1,
            prediction_label = "Hired",
            explanation      = "Strong profile",
            fairness_result  = {"sensitive_attribute": "gender", "is_fair": True, "warning": None},
            correlation_id   = "corr-pred-001",
        )

    def test_log_prediction_scrubs_explanation(self, tmp_audit, monkeypatch, caplog):
        """PII in explanation text must be redacted before logging."""
        monkeypatch.setenv("PII_MASK_ENABLED", "true")
        import importlib
        import utils.logger as log_mod
        importlib.reload(log_mod)

        import logging
        with caplog.at_level(logging.INFO, logger="predictions"):
            from utils.logger import log_prediction
            log_prediction(
                domain           = "hiring",
                input_data       = {"technical_score": 80},
                prediction       = 1,
                prediction_label = "Hired",
                explanation      = "Candidate: Alice Smith, email alice@corp.com hired",
                fairness_result  = {"sensitive_attribute": "gender", "is_fair": True, "warning": None},
            )
        combined = " ".join(caplog.messages)
        assert "alice@corp.com" not in combined


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE — JSON fallback (no MongoDB)
# ═════════════════════════════════════════════════════════════════════════════

class TestDatabaseJsonFallback:
    """All tests run without MongoDB (MONGO_URI unset → get_database() returns None)."""

    @pytest.fixture(autouse=True)
    def no_mongo(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MONGO_URI", raising=False)
        # Point JSON log to temp dir
        import importlib
        import utils.database as db_mod
        db_mod.JSON_LOG_PATH = tmp_path / "predictions.json"
        db_mod._client = None
        db_mod._db     = None
        self.db_mod  = db_mod
        self.json_path = db_mod.JSON_LOG_PATH

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_save_prediction_creates_json(self):
        record = {
            "domain": "hiring", "prediction": 1,
            "prediction_label": "Hired", "correlation_id": "abc",
        }
        self._run(self.db_mod.save_prediction(record))
        assert self.json_path.exists()

    def test_save_prediction_adds_timestamp(self):
        record = {"domain": "loan", "prediction": 0}
        self._run(self.db_mod.save_prediction(record))
        saved = json.loads(self.json_path.read_text())[0]
        assert "timestamp" in saved

    def test_multiple_saves_all_persisted(self):
        for i in range(5):
            self._run(self.db_mod.save_prediction({"domain": "hiring", "idx": i}))
        records = json.loads(self.json_path.read_text())
        assert len(records) == 5

    def test_get_recent_returns_empty_if_no_file(self):
        if self.json_path.exists():
            self.json_path.unlink()
        result = self._run(self.db_mod.get_recent_predictions("hiring", limit=10))
        assert result == []

    def test_get_recent_filters_by_domain(self):
        for domain in ("hiring", "loan", "hiring", "social"):
            self._run(self.db_mod.save_prediction({"domain": domain, "prediction": 1}))
        result = self._run(self.db_mod.get_recent_predictions("hiring", limit=100))
        assert all(r.get("domain") == "hiring" for r in result)

    def test_get_recent_respects_limit(self):
        for i in range(20):
            self._run(self.db_mod.save_prediction({"domain": "loan", "prediction": i % 2}))
        result = self._run(self.db_mod.get_recent_predictions("loan", limit=5))
        assert len(result) <= 5

    def test_get_recent_filters_by_sensitive_attr(self):
        records = [
            {"domain": "hiring", "fairness": {"sensitive_attribute": "gender"},   "prediction": 1},
            {"domain": "hiring", "fairness": {"sensitive_attribute": "ethnicity"}, "prediction": 0},
        ]
        for r in records:
            self._run(self.db_mod.save_prediction(r))

        result = self._run(
            self.db_mod.get_recent_predictions("hiring", limit=100, sensitive_attr="gender")
        )
        assert all(
            r.get("fairness", {}).get("sensitive_attribute") == "gender"
            for r in result
        )

    def test_corrupted_json_fallback_returns_empty(self):
        self.json_path.write_text("{{invalid json}}")
        result = self._run(self.db_mod.get_recent_predictions("hiring"))
        assert result == []


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE — MongoDB path (fully mocked)
# ═════════════════════════════════════════════════════════════════════════════

class TestDatabaseMongoPath:

    @pytest.fixture(autouse=True)
    def mock_mongo(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MONGO_URI", "mongodb://localhost/fake")

        self.mock_collection = AsyncMock()
        self.mock_collection.insert_one = AsyncMock()

        mock_cursor = MagicMock()
        mock_cursor.sort   = MagicMock(return_value=mock_cursor)
        mock_cursor.limit  = MagicMock(return_value=mock_cursor)
        mock_cursor.allow_disk_use = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=[
            {"domain": "hiring", "prediction": 1, "fairness": {}}
        ])
        self.mock_collection.find = MagicMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=self.mock_collection)

        import utils.database as db_mod
        db_mod._client = MagicMock()
        db_mod._db     = mock_db
        self.db_mod  = db_mod

        monkeypatch.setattr(db_mod, "MOTOR_AVAILABLE", True)
        monkeypatch.setattr(db_mod, "get_database",    lambda: mock_db)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_save_prediction_calls_insert_one(self):
        self._run(self.db_mod.save_prediction({"domain": "hiring", "prediction": 1}))
        self.mock_collection.insert_one.assert_called_once()

    def test_get_recent_calls_find(self):
        self._run(self.db_mod.get_recent_predictions("hiring", limit=50))
        self.mock_collection.find.assert_called()

    def test_mongo_write_failure_falls_back_to_json(self, tmp_path):
        self.mock_collection.insert_one.side_effect = Exception("connection lost")
        import utils.database as db_mod
        db_mod.JSON_LOG_PATH = tmp_path / "fallback.json"
        self._run(db_mod.save_prediction({"domain": "loan", "prediction": 0}))
        assert db_mod.JSON_LOG_PATH.exists()


# ═════════════════════════════════════════════════════════════════════════════
# ensure_indexes — no-op without Mongo
# ═════════════════════════════════════════════════════════════════════════════

class TestEnsureIndexes:

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_ensure_indexes_no_mongo_no_crash(self, monkeypatch):
        monkeypatch.delenv("MONGO_URI", raising=False)
        import utils.database as db_mod
        db_mod._client = None
        db_mod._db     = None
        # Should complete without raising
        self._run(db_mod.ensure_indexes())

    def test_ensure_indexes_mongo_error_no_crash(self, monkeypatch):
        """Index creation errors must be caught, not propagated."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        import utils.database as db_mod

        broken_collection = AsyncMock()
        broken_collection.create_index = AsyncMock(side_effect=Exception("timeout"))
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=broken_collection)

        monkeypatch.setattr(db_mod, "get_database",    lambda: mock_db)
        monkeypatch.setattr(db_mod, "MOTOR_AVAILABLE", True)

        self._run(db_mod.ensure_indexes())   # must not raise
