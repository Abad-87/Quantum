"""
tests/conftest.py

Shared pytest fixtures for the full Quantum test suite.

Fixture hierarchy
-----------------
dummy_models_dir   — creates real RandomForest .pkl files in a tmp dir
mock_registry      — injects those models into the singleton ModelRegistry
app_client         — TestClient wrapping the FastAPI app with models loaded
mock_db            — patches save_prediction / get_recent_predictions to no-ops

Every test file imports what it needs from here via pytest's auto-discovery.
No test file sets sys.path — conftest.py at the package root handles that.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import joblib
import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

# ── Ensure project root is importable ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Disable PII masking during tests (keeps logs readable) ───────────────────
os.environ.setdefault("PII_MASK_ENABLED", "false")
# ── Point to a scratch log dir so tests never write to repo logs/ ────────────
_TMP_LOGS = tempfile.mkdtemp(prefix="quantum_test_logs_")
os.environ.setdefault("AUDIT_LOG_DIR", _TMP_LOGS)


# ═════════════════════════════════════════════════════════════════════════════
# CANONICAL PAYLOADS  (reused by all test modules)
# ═════════════════════════════════════════════════════════════════════════════

HIRING_PAYLOAD: dict = {
    "years_experience":     5,
    "education_level":      2,
    "technical_score":      82,
    "communication_score":  75,
    "num_past_jobs":        3,
    "certifications":       2,
    "gender":               "female",
}

LOAN_PAYLOAD: dict = {
    "credit_score":     720,
    "annual_income":    75_000,
    "loan_amount":      25_000,
    "loan_term_months": 36,
    "employment_years": 4,
    "existing_debt":    8_000,
    "num_credit_lines": 3,
    "ethnicity":        "hispanic",
}

SOCIAL_PAYLOAD: dict = {
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


# ═════════════════════════════════════════════════════════════════════════════
# DUMMY MODEL FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def _make_hiring_model(path: Path) -> RandomForestClassifier:
    np.random.seed(0)
    X = np.random.rand(200, 6) * [20, 3, 100, 100, 10, 5]
    y = (X[:, 2] + X[:, 3] > 100).astype(int)
    m = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    joblib.dump(m, path)
    return m


def _make_loan_model(path: Path) -> RandomForestClassifier:
    np.random.seed(1)
    X = np.column_stack([
        np.random.randint(300, 850, 200),
        np.random.randint(20_000, 200_000, 200),
        np.random.randint(1_000, 100_000, 200),
        np.random.choice([12, 24, 36, 60], 200),
        np.random.randint(0, 20, 200),
        np.random.randint(0, 50_000, 200),
        np.random.randint(0, 10, 200),
    ])
    dti = X[:, 5] / (X[:, 1] + 1)
    y   = ((X[:, 0] > 620) & (dti < 0.5)).astype(int)
    m   = RandomForestClassifier(n_estimators=10, random_state=1).fit(X, y)
    joblib.dump(m, path)
    return m


def _make_social_model(path: Path) -> RandomForestClassifier:
    np.random.seed(2)
    X = np.column_stack([
        np.random.rand(200) * 120,
        np.random.rand(200) * 10,
        np.random.randint(1, 30, 200),
        np.random.rand(200),
        np.random.rand(200) * 0.5,
        np.random.rand(200) * 0.3,
        np.random.randint(1, 2_000, 200),
    ])
    y = np.random.randint(0, 8, 200)
    m = RandomForestClassifier(n_estimators=10, random_state=2).fit(X, y)
    joblib.dump(m, path)
    return m


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def dummy_models_dir() -> Generator[Path, None, None]:
    """
    Session-scoped: build real .pkl files once and reuse across all tests.
    Returns the directory Path containing:
      hiring_model.pkl / loan_model.pkl / social_model.pkl
    """
    with tempfile.TemporaryDirectory(prefix="quantum_models_") as td:
        d = Path(td)
        _make_hiring_model(d / "hiring_model.pkl")
        _make_loan_model  (d / "loan_model.pkl")
        _make_social_model(d / "social_model.pkl")
        yield d


@pytest.fixture(scope="session")
def loaded_models(dummy_models_dir: Path) -> dict:
    """Return {name: sklearn model object} loaded directly with joblib."""
    return {
        "hiring": joblib.load(dummy_models_dir / "hiring_model.pkl"),
        "loan":   joblib.load(dummy_models_dir / "loan_model.pkl"),
        "social": joblib.load(dummy_models_dir / "social_model.pkl"),
    }


@pytest.fixture(scope="session")
def mock_registry(dummy_models_dir: Path):
    """
    Inject dummy .pkl files into the ModelRegistry singleton before any
    test in the session imports the FastAPI app.  The registry is patched
    at the module level so all routers see it automatically.
    """
    from utils.model_registry import registry

    registry.load("hiring", dummy_models_dir / "hiring_model.pkl")
    registry.load("loan",   dummy_models_dir / "loan_model.pkl")
    registry.load("social", dummy_models_dir / "social_model.pkl")
    return registry


@pytest.fixture(scope="session")
def app_client(mock_registry) -> Generator[TestClient, None, None]:
    """
    Session-scoped TestClient.  Models are pre-loaded into the registry
    so the lifespan hook does not attempt to read from disk paths that may
    not match the repo layout in CI.

    Database writes and index creation are mocked so the client never needs
    a real MongoDB instance.
    """
    async def _noop_ensure_indexes():
        pass

    async def _noop_save_prediction(record: dict):
        pass

    async def _noop_get_recent(domain, limit=100, sensitive_attr=None, projection=None):
        return []

    with (
        patch("utils.database.ensure_indexes",          new=_noop_ensure_indexes),
        patch("utils.database.save_prediction",         new=_noop_save_prediction),
        patch("utils.database.get_recent_predictions",  new=_noop_get_recent),
        patch("hiring.model_loader.preload",            return_value=None),
        patch("loan.model_loader.preload",              return_value=None),
        patch("social.model_loader.preload",            return_value=None),
    ):
        from main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


@pytest.fixture()
def mock_db():
    """
    Function-scoped: patch all database I/O for unit tests that don't
    need the full HTTP stack.
    """
    with (
        patch("utils.database.save_prediction",        new=AsyncMock()),
        patch("utils.database.get_recent_predictions", new=AsyncMock(return_value=[])),
    ):
        yield


@pytest.fixture()
def mock_model():
    """
    Lightweight fake sklearn model.  predict() always returns [1];
    predict_proba() returns [[0.1, 0.9]].
    """
    m = MagicMock()
    m.predict       = MagicMock(return_value=np.array([1]))
    m.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))
    return m
