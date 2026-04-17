"""
tests/test_model_loading.py

Tests for the ModelRegistry singleton and each domain's model_loader adapter.

Coverage targets
----------------
- Happy-path load (real .pkl)
- Idempotent load (hash check skips duplicate disk read)
- Missing file → FileNotFoundError with helpful message
- Corrupted file → exception propagated cleanly
- hot_swap() atomically replaces a cached model
- A/B routing: register_ab_variant + get_ab traffic split
- list_models() returns correct provenance metadata
- SHA-256 version fingerprint changes when file changes
"""

from __future__ import annotations

import io
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from utils.model_registry import ModelRegistry


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _tiny_model(path: Path, seed: int = 42) -> RandomForestClassifier:
    """Train and save a 5-estimator RF to *path*.  Returns the model object."""
    np.random.seed(seed)
    X = np.random.rand(40, 3)
    y = (X[:, 0] > 0.5).astype(int)
    m = RandomForestClassifier(n_estimators=5, random_state=seed).fit(X, y)
    joblib.dump(m, path)
    return m


def _corrupt_file(path: Path) -> None:
    """Write random bytes to *path* so joblib cannot deserialise it."""
    path.write_bytes(b"\xde\xad\xbe\xef" * 64)


# ═════════════════════════════════════════════════════════════════════════════
# Fresh registry per test (not the singleton)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def reg() -> ModelRegistry:
    """Return a fresh, empty ModelRegistry instance for each test."""
    return ModelRegistry()


# ═════════════════════════════════════════════════════════════════════════════
# Happy-path loading
# ═════════════════════════════════════════════════════════════════════════════

class TestHappyPathLoad:

    def test_load_returns_model_entry(self, reg, tmp_path):
        p = tmp_path / "model.pkl"
        _tiny_model(p)
        entry = reg.load("test", p)

        assert entry.name    == "test"
        assert entry.variant == "primary"
        assert len(entry.short_version) == 12
        assert entry.loaded_at            # ISO timestamp not empty

    def test_get_returns_sklearn_object(self, reg, tmp_path):
        p = tmp_path / "model.pkl"
        _tiny_model(p)
        reg.load("test", p)
        model = reg.get("test")

        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_predict_works(self, reg, tmp_path):
        p = tmp_path / "model.pkl"
        _tiny_model(p)
        reg.load("test", p)
        model = reg.get("test")

        X = np.random.rand(1, 3)
        pred = model.predict(X)
        assert pred.shape == (1,)
        assert pred[0] in {0, 1}

    def test_idempotent_load_skips_disk_read(self, reg, tmp_path):
        """Loading the same file twice must not call joblib.load twice."""
        p = tmp_path / "model.pkl"
        _tiny_model(p)
        reg.load("test", p)

        with patch("joblib.load") as mock_load:
            reg.load("test", p)          # same path, same hash → cache hit
            mock_load.assert_not_called()

    def test_load_all_convenience(self, reg, tmp_path):
        p1 = tmp_path / "a.pkl"
        p2 = tmp_path / "b.pkl"
        _tiny_model(p1, seed=1)
        _tiny_model(p2, seed=2)

        reg.load_all({"alpha": p1, "beta": p2})

        assert reg.get("alpha") is not None
        assert reg.get("beta")  is not None

    def test_real_dummy_models_load(self, dummy_models_dir):
        """Integration test: load the real dummy .pkl files created by conftest."""
        r = ModelRegistry()
        r.load("hiring", dummy_models_dir / "hiring_model.pkl")
        r.load("loan",   dummy_models_dir / "loan_model.pkl")
        r.load("social", dummy_models_dir / "social_model.pkl")

        for name in ("hiring", "loan", "social"):
            assert r.get(name) is not None
            assert len(r.get_version(name)) == 12


# ═════════════════════════════════════════════════════════════════════════════
# Error conditions
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorConditions:

    def test_missing_file_raises_file_not_found(self, reg, tmp_path):
        missing = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError) as exc_info:
            reg.load("x", missing)
        assert "does_not_exist.pkl" in str(exc_info.value)

    def test_corrupted_file_raises_exception(self, reg, tmp_path):
        p = tmp_path / "corrupt.pkl"
        _corrupt_file(p)
        with pytest.raises(Exception):    # joblib raises various errors on bad bytes
            reg.load("x", p)

    def test_get_unloaded_model_raises_key_error(self, reg):
        with pytest.raises(KeyError) as exc_info:
            reg.get("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_get_missing_variant_raises_key_error(self, reg, tmp_path):
        p = tmp_path / "m.pkl"
        _tiny_model(p)
        reg.load("x", p, variant="primary")
        with pytest.raises(KeyError):
            reg.get("x", variant="challenger")

    def test_unknown_version_returns_string(self, reg):
        v = reg.get_version("not_loaded")
        assert v == "unknown"


# ═════════════════════════════════════════════════════════════════════════════
# Hot-swap
# ═════════════════════════════════════════════════════════════════════════════

class TestHotSwap:

    def test_hot_swap_replaces_model(self, reg, tmp_path):
        p1 = tmp_path / "v1.pkl"
        p2 = tmp_path / "v2.pkl"
        _tiny_model(p1, seed=1)
        _tiny_model(p2, seed=99)           # different seed → different model

        reg.load("m", p1)
        v1 = reg.get_version("m")

        reg.hot_swap("m", p2)
        v2 = reg.get_version("m")

        assert v1 != v2                    # version fingerprint changed
        assert reg.get_metadata("m")["path"] == str(p2)

    def test_hot_swap_missing_target_raises(self, reg, tmp_path):
        p = tmp_path / "original.pkl"
        _tiny_model(p)
        reg.load("m", p)

        with pytest.raises(FileNotFoundError):
            reg.hot_swap("m", tmp_path / "ghost.pkl")

    def test_sha256_changes_when_content_changes(self, reg, tmp_path):
        p1 = tmp_path / "model_a.pkl"
        p2 = tmp_path / "model_b.pkl"
        _tiny_model(p1, seed=10)
        _tiny_model(p2, seed=20)

        reg.load("q", p1)
        version_a = reg.get_version("q")

        reg.hot_swap("q", p2)
        version_b = reg.get_version("q")

        assert version_a != version_b


# ═════════════════════════════════════════════════════════════════════════════
# A/B routing
# ═════════════════════════════════════════════════════════════════════════════

class TestABRouting:

    def test_register_ab_variant(self, reg, tmp_path):
        p_primary    = tmp_path / "primary.pkl"
        p_challenger = tmp_path / "challenger.pkl"
        _tiny_model(p_primary,    seed=1)
        _tiny_model(p_challenger, seed=2)

        reg.load("dom", p_primary)
        reg.register_ab_variant("dom", "challenger", p_challenger, traffic_fraction=0.20)

        # Both variants must be accessible by name
        assert reg.get("dom", "primary")    is not None
        assert reg.get("dom", "challenger") is not None

    def test_get_ab_returns_tuple(self, reg, tmp_path):
        p = tmp_path / "m.pkl"
        _tiny_model(p)
        reg.load("dom", p)

        model, variant = reg.get_ab("dom")
        assert variant  == "primary"        # no split → always primary
        assert model    is not None

    def test_ab_traffic_split_coverage(self, reg, tmp_path):
        """With 50 % split, over 2 000 draws both variants must appear."""
        p1 = tmp_path / "p.pkl"
        p2 = tmp_path / "c.pkl"
        _tiny_model(p1, seed=1)
        _tiny_model(p2, seed=2)

        reg.load("dom", p1)
        reg.register_ab_variant("dom", "challenger", p2, traffic_fraction=0.5)

        variants_seen = set()
        for _ in range(2_000):
            _, v = reg.get_ab("dom")
            variants_seen.add(v)

        assert "primary"    in variants_seen
        assert "challenger" in variants_seen

    def test_clear_ab_split_reverts_to_primary(self, reg, tmp_path):
        p1 = tmp_path / "p.pkl"
        p2 = tmp_path / "c.pkl"
        _tiny_model(p1)
        _tiny_model(p2)

        reg.load("dom", p1)
        reg.register_ab_variant("dom", "challenger", p2, traffic_fraction=0.5)
        reg.clear_ab_split("dom")

        for _ in range(50):
            _, v = reg.get_ab("dom")
            assert v == "primary"

    def test_invalid_traffic_fraction_raises(self, reg, tmp_path):
        p = tmp_path / "m.pkl"
        _tiny_model(p)
        reg.load("dom", p)

        with pytest.raises(ValueError):
            reg.register_ab_variant("dom", "c", p, traffic_fraction=0.0)

        with pytest.raises(ValueError):
            reg.register_ab_variant("dom", "c", p, traffic_fraction=1.0)


# ═════════════════════════════════════════════════════════════════════════════
# list_models / metadata
# ═════════════════════════════════════════════════════════════════════════════

class TestMetadata:

    def test_list_models_structure(self, reg, tmp_path):
        p = tmp_path / "m.pkl"
        _tiny_model(p)
        reg.load("hiring", p)

        snapshot = reg.list_models()
        assert "hiring" in snapshot
        assert "primary" in snapshot["hiring"]
        entry = snapshot["hiring"]["primary"]
        assert "version"   in entry
        assert "loaded_at" in entry
        assert "path"      in entry

    def test_get_metadata_full(self, reg, tmp_path):
        p = tmp_path / "m.pkl"
        _tiny_model(p)
        reg.load("hiring", p)

        meta = reg.get_metadata("hiring")
        assert meta["name"]    == "hiring"
        assert meta["variant"] == "primary"
        assert len(meta["version"]) == 12

    def test_get_metadata_unloaded(self, reg):
        meta = reg.get_metadata("ghost")
        assert meta["status"] == "not_loaded"


# ═════════════════════════════════════════════════════════════════════════════
# Domain model_loader adapters
# ═════════════════════════════════════════════════════════════════════════════

class TestDomainLoaders:
    """
    Test the thin adapter layer (hiring/model_loader.py etc.).
    These delegate to the global registry singleton, so we patch it.
    """

    def test_hiring_loader_preload(self, dummy_models_dir):
        import hiring.model_loader as hl
        from utils.model_registry import registry as _reg

        _reg.load("hiring", dummy_models_dir / "hiring_model.pkl")
        model = hl.get_model()
        assert hasattr(model, "predict")

    def test_loan_loader_preload(self, dummy_models_dir):
        import loan.model_loader as ll
        from utils.model_registry import registry as _reg

        _reg.load("loan", dummy_models_dir / "loan_model.pkl")
        model = ll.get_model()
        assert hasattr(model, "predict")

    def test_social_loader_preload(self, dummy_models_dir):
        import social.model_loader as sl
        from utils.model_registry import registry as _reg

        _reg.load("social", dummy_models_dir / "social_model.pkl")
        model = sl.get_model()
        assert hasattr(model, "predict")

    def test_get_version_is_12_chars(self, dummy_models_dir):
        import hiring.model_loader as hl
        from utils.model_registry import registry as _reg

        _reg.load("hiring", dummy_models_dir / "hiring_model.pkl")
        v = hl.get_version()
        assert isinstance(v, str)
        assert len(v) == 12

    def test_hot_swap_via_loader(self, tmp_path, dummy_models_dir):
        import hiring.model_loader as hl
        from utils.model_registry import registry as _reg

        _reg.load("hiring", dummy_models_dir / "hiring_model.pkl")
        v1 = hl.get_version()

        new_path = tmp_path / "hiring_v2.pkl"
        _tiny_model(new_path, seed=77)
        hl.hot_swap(new_path)

        v2 = hl.get_version()
        assert v1 != v2
