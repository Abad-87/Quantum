"""
utils/model_registry.py

Centralized Model Registry — single source of truth for all .pkl models.

Responsibilities:
  • Load every model exactly once at application startup (not per-request).
  • Cache models in memory behind a thread-safe RLock so concurrent requests
    never block each other or trigger duplicate disk reads.
  • Version every model via the SHA-256 hash of its .pkl file so audit logs
    can pinpoint the exact artifact used for any prediction.
  • Hot-swap a model from a new file path at runtime with zero downtime —
    the swap is atomic: in-flight requests finish on the old model, new
    requests get the replacement the instant the lock is released.
  • Support A/B fairness testing by routing a configurable traffic fraction
    to an alternative model variant without touching the primary slot.

Public interface (all thread-safe):
  registry.load(name, path, variant="primary")
  registry.load_all({"hiring": Path(...), ...})
  registry.get(name, variant="primary")          → model object
  registry.get_ab(name)                          → (model, variant_name)
  registry.get_version(name, variant)            → "abc123def456" (12-char hex)
  registry.get_metadata(name, variant)           → dict
  registry.hot_swap(name, new_path, variant)
  registry.register_ab_variant(name, variant, path, traffic_fraction)
  registry.clear_ab_split(name)
  registry.list_models()                         → nested dict of metadata
"""

import hashlib
import logging
import random
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

logger = logging.getLogger("model_registry")


# ─── Model record ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelEntry:
    """Immutable snapshot of a loaded model and its provenance."""
    model:     Any        # The deserialized sklearn object
    path:      Path       # Source file path
    version:   str        # Full SHA-256 hex digest of the .pkl file
    loaded_at: str        # ISO-8601 UTC timestamp
    name:      str        # Logical domain name (e.g. "hiring")
    variant:   str        # Slot name within that domain ("primary", "variant_a", …)

    @property
    def short_version(self) -> str:
        """First 12 hex chars — compact enough for log lines."""
        return self.version[:12]


# ─── Registry ────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Thread-safe in-memory model cache with hot-swap and A/B routing.

    Internal layout
    ---------------
    _models:         {domain_name: {variant_name: ModelEntry}}
    _traffic_splits: {domain_name: {variant_name: weight}}   (sum-to-1 floats)
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._models:         Dict[str, Dict[str, ModelEntry]] = {}
        self._traffic_splits: Dict[str, Dict[str, float]]    = {}

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self, name: str, path: Path, variant: str = "primary") -> ModelEntry:
        """
        Load a model from *path* and cache it under (*name*, *variant*).

        Idempotent: if the file hash matches the already-cached entry, the
        existing entry is returned immediately without re-reading the file.

        Raises FileNotFoundError if the .pkl is missing.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Model '{name}/{variant}' not found at '{path}'. "
                "Run  python create_dummy_models.py  to create test fixtures."
            )

        version = self._sha256(path)

        with self._lock:
            existing = self._models.get(name, {}).get(variant)
            if existing and existing.version == version:
                logger.debug(
                    f"'{name}/{variant}' already current (v{existing.short_version}). "
                    "Skipping disk read."
                )
                return existing

            logger.info(f"Loading '{name}/{variant}' from {path} …")
            model_obj = joblib.load(path)

            entry = ModelEntry(
                model=model_obj,
                path=path,
                version=version,
                loaded_at=datetime.now(timezone.utc).isoformat(),
                name=name,
                variant=variant,
            )

            if name not in self._models:
                self._models[name] = {}
            self._models[name][variant] = entry

            logger.info(
                f"Loaded '{name}/{variant}'  version={entry.short_version}  "
                f"path={path}"
            )
            return entry

    def load_all(self, model_paths: Dict[str, Path]) -> None:
        """
        Convenience startup helper — load every domain model in one call.

        Example::

            registry.load_all({
                "hiring": Path("models/hiring_model.pkl"),
                "loan":   Path("models/loan_model.pkl"),
                "social": Path("models/social_model.pkl"),
            })
        """
        for name, path in model_paths.items():
            self.load(name, path)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def get(self, name: str, variant: str = "primary") -> Any:
        """
        Return the cached model object.  O(1), thread-safe.

        Raises KeyError if the model has not been loaded yet.
        """
        with self._lock:
            entry = self._models.get(name, {}).get(variant)
            if entry is None:
                raise KeyError(
                    f"Model '{name}/{variant}' is not loaded. "
                    "Call registry.load() or registry.load_all() at startup."
                )
            return entry.model

    def get_ab(self, name: str) -> Tuple[Any, str]:
        """
        Return *(model, variant_name)* using the configured traffic split.

        Falls back to ("primary", model) when no A/B split is registered.
        The selection is weighted-random and stateless — no sticky sessions.
        """
        with self._lock:
            splits = self._traffic_splits.get(name)
            if not splits:
                return self.get(name, "primary"), "primary"

            variants = list(splits.keys())
            weights  = [splits[v] for v in variants]
            chosen   = random.choices(variants, weights=weights, k=1)[0]
            return self.get(name, chosen), chosen

    def get_version(self, name: str, variant: str = "primary") -> str:
        """Return the 12-char version hex, or 'unknown' if not loaded."""
        with self._lock:
            entry = self._models.get(name, {}).get(variant)
            return entry.short_version if entry else "unknown"

    def get_metadata(self, name: str, variant: str = "primary") -> dict:
        """Return provenance metadata for audit-log embedding."""
        with self._lock:
            entry = self._models.get(name, {}).get(variant)
            if entry is None:
                return {"name": name, "variant": variant, "status": "not_loaded"}
            return {
                "name":      entry.name,
                "variant":   entry.variant,
                "version":   entry.short_version,
                "path":      str(entry.path),
                "loaded_at": entry.loaded_at,
            }

    # ── Hot-swap ─────────────────────────────────────────────────────────────

    def hot_swap(
        self,
        name:    str,
        new_path: Path,
        variant: str = "primary",
    ) -> ModelEntry:
        """
        Atomically replace a cached model with the contents of *new_path*.

        - Requests in-flight finish on the old model (the lock is held only
          during the dict update, not during the potentially-slow joblib.load).
        - The new ModelEntry is constructed off-lock, then swapped in under the
          lock, so the window during which the lock is held is O(μs).

        Usage::

            registry.hot_swap("hiring", Path("models/hiring_model_v2.pkl"))
        """
        logger.info(f"Hot-swap initiated: '{name}/{variant}' ← {new_path}")

        if not new_path.exists():
            raise FileNotFoundError(f"Swap target not found: {new_path}")

        version   = self._sha256(new_path)
        model_obj = joblib.load(new_path)          # ← load off-lock (slow I/O)

        new_entry = ModelEntry(
            model=model_obj,
            path=new_path,
            version=version,
            loaded_at=datetime.now(timezone.utc).isoformat(),
            name=name,
            variant=variant,
        )

        with self._lock:                           # ← atomic swap (fast)
            if name not in self._models:
                self._models[name] = {}
            self._models[name][variant] = new_entry

        logger.info(
            f"Hot-swap complete: '{name}/{variant}' → v{new_entry.short_version}"
        )
        return new_entry

    # ── A/B Testing ──────────────────────────────────────────────────────────

    def register_ab_variant(
        self,
        name:             str,
        variant_name:     str,
        path:             Path,
        traffic_fraction: float = 0.1,
    ) -> None:
        """
        Register a challenger model and configure traffic routing.

        *traffic_fraction* of requests will be served by *variant_name*;
        the remainder continue to "primary".

        Example — send 10 % of hiring predictions to a re-trained model::

            registry.register_ab_variant(
                "hiring",
                "challenger",
                Path("models/hiring_model_v2.pkl"),
                traffic_fraction=0.10,
            )
        """
        if not 0.0 < traffic_fraction < 1.0:
            raise ValueError("traffic_fraction must be in the open interval (0, 1).")

        self.load(name, path, variant=variant_name)

        primary_fraction = round(1.0 - traffic_fraction, 6)
        with self._lock:
            self._traffic_splits[name] = {
                "primary":    primary_fraction,
                variant_name: round(traffic_fraction, 6),
            }

        logger.info(
            f"A/B split registered for '{name}': "
            f"primary={primary_fraction:.1%}, {variant_name}={traffic_fraction:.1%}"
        )

    def clear_ab_split(self, name: str) -> None:
        """Remove any A/B routing for *name* — all traffic reverts to 'primary'."""
        with self._lock:
            removed = self._traffic_splits.pop(name, None)
        if removed:
            logger.info(f"A/B split cleared for '{name}'. All traffic → primary.")

    # ── Inspection ───────────────────────────────────────────────────────────

    def list_models(self) -> dict:
        """Snapshot of all loaded models — safe to serialise to JSON."""
        with self._lock:
            return {
                domain: {
                    variant: {
                        "version":   e.short_version,
                        "loaded_at": e.loaded_at,
                        "path":      str(e.path),
                    }
                    for variant, e in variants.items()
                }
                for domain, variants in self._models.items()
            }

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _sha256(path: Path) -> str:
        """Streaming SHA-256 of a file — handles large .pkl files without OOM."""
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                digest.update(chunk)
        return digest.hexdigest()


# ─── Module-level singleton ───────────────────────────────────────────────────
# Import this object in every domain module instead of calling joblib directly.
# The singleton is created once when Python first imports this module.
registry = ModelRegistry()
