"""
Quantum — Unbiased AI Decision Platform
main.py  —  FastAPI application entry point  (Phase 3)

Phase 3 additions
-----------------
1. ensure_indexes() called at startup — creates all MongoDB compound indexes
   before the server begins accepting requests.

2. SHAP router mounted at /shap — provides:
   GET  /shap/{correlation_id}       poll for completed SHAP report
   WS   /shap/ws/{correlation_id}    push notification when report is ready

3. Version bumped to 3.0.0.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hiring.router   import router as hiring_router
from loan.router     import router as loan_router
from social.router   import router as social_router
from utils.shap_cache import router as shap_router
from utils.logger    import setup_logger, log_correlation_event
from utils.model_registry import registry
from utils.database  import ensure_indexes

import hiring.model_loader  as hiring_loader
import loan.model_loader    as loan_loader
import social.model_loader  as social_loader

logger = setup_logger("main")

_SENSITIVE_KEYS = frozenset({
    "gender", "religion", "ethnicity", "race",
    "age_group", "location", "language", "disability",
})


# ─── Lifespan: startup / shutdown ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
      1. Preload all domain models into the ModelRegistry (fail-fast).
      2. Ensure all MongoDB indexes exist (non-fatal if Mongo is absent).
    """
    logger.info("=== Quantum startup (Phase 3) ===")

    # 1. Models
    try:
        hiring_loader.preload()
        loan_loader.preload()
        social_loader.preload()
    except FileNotFoundError as exc:
        logger.error(
            f"FATAL: Model file missing — {exc}\n"
            "Run  python create_dummy_models.py  and restart."
        )
        raise

    # 2. MongoDB indexes
    await ensure_indexes()

    logger.info("=== All models loaded. Indexes verified. Server ready. ===")
    yield
    logger.info("=== Quantum shutdown ===")


# ─── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Quantum – Unbiased AI Decision Platform",
    description = (
        "Fairness-aware AI backend for Job Hiring, Loan Approval, and Social "
        "Recommendation.  SHAP explanations are computed asynchronously and "
        "delivered via GET /shap/{id} or WebSocket /shap/ws/{id}."
    ),
    version     = "3.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── Correlation middleware ───────────────────────────────────────────────────

@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    start_ms = time.monotonic()

    body_bytes: bytes = await request.body()
    raw_payload: Optional[dict] = None
    if body_bytes:
        try:
            raw_payload = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    correlation_id    = str(uuid.uuid4())
    path              = request.url.path
    domain            = _path_to_domain(path)
    model_meta        = registry.get_metadata(domain) if domain else {}
    sanitised_payload = _sanitise(raw_payload)

    if path not in ("/health", "/", "/docs", "/redoc", "/openapi.json"):
        log_correlation_event(
            correlation_id = correlation_id,
            event          = "request_received",
            path           = path,
            method         = request.method,
            payload        = sanitised_payload,
            model_metadata = model_meta,
            result         = None,
        )

    request.state.correlation_id = correlation_id
    request.state.domain         = domain

    response   = await call_next(request)
    elapsed_ms = round((time.monotonic() - start_ms) * 1000, 2)

    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time"]  = f"{elapsed_ms}ms"

    logger.info(
        f"{request.method} {path} → {response.status_code} "
        f"[{elapsed_ms}ms] corr={correlation_id[:8]}…"
    )
    return response

# ─── Global exception handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.error(f"[{correlation_id}] Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":          "Internal server error",
            "detail":         str(exc),
            "correlation_id": correlation_id,
        },
    )

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(hiring_router, prefix="/hiring", tags=["Hiring"])
app.include_router(loan_router,   prefix="/loan",   tags=["Loan"])
app.include_router(social_router, prefix="/social", tags=["Social"])
app.include_router(shap_router,                     tags=["SHAP"])

# ─── Platform endpoints ───────────────────────────────────────────────────────

@app.get("/", tags=["Platform"])
def root():
    return {
        "status":   "online",
        "platform": "Quantum – Unbiased AI Decision Platform",
        "version":  "3.0.0",
        "endpoints": [
            "POST /hiring/predict",
            "POST /loan/predict",
            "POST /social/recommend",
            "GET  /shap/{correlation_id}    — poll for SHAP report",
            "WS   /shap/ws/{correlation_id} — push when SHAP ready",
            "GET  /models",
            "GET  /health",
            "GET  /docs",
        ],
    }


@app.get("/health", tags=["Platform"])
def health_check():
    from utils.shap_cache import shap_cache, ws_manager
    audit_path = Path("logs/audit.jsonl")
    return {
        "status":    "healthy",
        "timestamp": time.time(),
        "models":    registry.list_models(),
        "audit_log": {
            "path":       str(audit_path),
            "exists":     audit_path.exists(),
            "size_bytes": audit_path.stat().st_size if audit_path.exists() else 0,
        },
        "shap_cache": {
            "backend":        "redis+memory" if shap_cache._redis else "memory-only",
            "ws_connections": ws_manager.connected_count(),
        },
    }


@app.get("/models", tags=["Platform"])
def list_models():
    return {"models": registry.list_models(), "timestamp": time.time()}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _path_to_domain(path: str) -> Optional[str]:
    if path.startswith("/hiring"): return "hiring"
    if path.startswith("/loan"):   return "loan"
    if path.startswith("/social"): return "social"
    return None

def _sanitise(payload: Optional[dict]) -> Optional[dict]:
    if payload is None:
        return None
    return {k: v for k, v in payload.items() if k not in _SENSITIVE_KEYS}
