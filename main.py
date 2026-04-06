"""
Quantum — Unbiased AI Decision Platform
main.py  —  FastAPI application entry point

What's new in this revision
---------------------------
1. Lifespan hook — preloads all three models into the ModelRegistry at startup.
   The registry holds them in memory for the process lifetime.  No model is
   ever loaded on a per-request basis.

2. Correlation middleware — runs before every request:
   a. Reads and caches the raw request body (Starlette caches it on _body so
      the route handler can still read it normally).
   b. Generates a UUID-4 correlation_id.
   c. Derives which model is being used from the URL path.
   d. Writes an immutable "request_received" audit record to logs/audit.jsonl
      BEFORE the request is processed — the record includes the sanitised
      payload and model version so every prediction is traceable.
   e. Attaches correlation_id to request.state so routers can thread it
      through their own log/audit calls.
   f. Adds X-Correlation-ID response header for client-side tracing.

3. Enhanced /health endpoint — includes registry snapshot (model versions
   and load times) and audit-log file size.

4. New GET /models endpoint — machine-readable registry dump for ops tooling.
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
from utils.logger    import setup_logger, log_correlation_event
from utils.model_registry import registry

# ── Domain model loaders (needed only for preload() calls) ───────────────────
import hiring.model_loader  as hiring_loader
import loan.model_loader    as loan_loader
import social.model_loader  as social_loader

logger = setup_logger("main")

# ─── Sensitive keys that must never appear in the audit-log payload ───────────
_SENSITIVE_KEYS = frozenset({
    "gender", "religion", "ethnicity", "race",
    "age_group", "location", "language", "disability",
})


# ─── Lifespan: startup / shutdown ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs at process startup (before accepting requests) and shutdown.

    All three domain models are loaded here exactly once into the shared
    ModelRegistry.  If any .pkl file is missing, the server refuses to
    start — fail-fast is safer than serving requests with no model.
    """
    logger.info("=== Quantum startup: loading models ===")
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

    logger.info("=== All models loaded. Server ready. ===")
    yield
    logger.info("=== Quantum shutdown ===")


# ─── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Quantum – Unbiased AI Decision Platform",
    description = (
        "A fairness-aware AI backend for Job Hiring, Loan Approval, "
        "and Social Media Recommendation. "
        "Every request is assigned a correlation_id for end-to-end traceability."
    ),
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)


# ─── CORS ────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Restrict to your frontend URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Correlation middleware ───────────────────────────────────────────────────

@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    """
    Pre-request middleware — runs for every inbound HTTP call.

    Execution order
    ---------------
    1. Read and cache the request body (allows downstream handlers to re-read).
    2. Generate a UUID-4 correlation_id.
    3. Resolve which model version is being used (from the URL path).
    4. Sanitise the payload (strip sensitive keys).
    5. Write an immutable "request_received" record to audit.jsonl.
    6. Attach correlation_id to request.state for routers.
    7. Forward the request to the actual route handler.
    8. Annotate the response with X-Correlation-ID header.
    """
    start_ms = time.monotonic()

    # ── 1. Read + cache body ─────────────────────────────────────────────────
    # Starlette stores the bytes on request._body after the first read, so
    # subsequent await request.body() calls in the route handler return the
    # cached bytes — no double-read issue.
    body_bytes: bytes = await request.body()

    raw_payload: Optional[dict] = None
    if body_bytes:
        try:
            raw_payload = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass   # Non-JSON body (e.g. form data) — leave as None

    # ── 2. Correlation ID ─────────────────────────────────────────────────────
    correlation_id = str(uuid.uuid4())

    # ── 3. Resolve model metadata ──────────────────────────────────────────────
    # Map URL path prefix → domain name so we can embed the model version.
    path       = request.url.path
    domain     = _path_to_domain(path)
    model_meta = registry.get_metadata(domain) if domain else {}

    # ── 4. Sanitise payload ───────────────────────────────────────────────────
    # Sensitive keys are stripped so the audit log never contains PII.
    sanitised_payload = _sanitise(raw_payload)

    # ── 5. Write immutable pre-request audit record ───────────────────────────
    # This is synchronous and blocks briefly — intentional: the record must be
    # durable before we process the request so nothing is silently lost.
    if path not in ("/health", "/", "/docs", "/redoc", "/openapi.json"):
        log_correlation_event(
            correlation_id  = correlation_id,
            event           = "request_received",
            path            = path,
            method          = request.method,
            payload         = sanitised_payload,
            model_metadata  = model_meta,
            result          = None,
        )

    # ── 6. Attach to request state for downstream routers ────────────────────
    request.state.correlation_id = correlation_id
    request.state.domain         = domain

    # ── 7. Process request ────────────────────────────────────────────────────
    response = await call_next(request)

    # ── 8. Annotate response ──────────────────────────────────────────────────
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


# ─── Platform endpoints ───────────────────────────────────────────────────────

@app.get("/", tags=["Platform"])
def root():
    return {
        "status":   "online",
        "platform": "Quantum – Unbiased AI Decision Platform",
        "version":  "2.0.0",
        "endpoints": [
            "POST /hiring/predict",
            "POST /loan/predict",
            "POST /social/recommend",
            "GET  /models",
            "GET  /health",
            "GET  /docs",
        ],
    }


@app.get("/health", tags=["Platform"])
def health_check():
    """
    Extended health check — includes model registry snapshot and audit-log info.
    """
    audit_path = Path("logs/audit.jsonl")
    audit_info = {
        "path":        str(audit_path),
        "exists":      audit_path.exists(),
        "size_bytes":  audit_path.stat().st_size if audit_path.exists() else 0,
    }
    return {
        "status":      "healthy",
        "timestamp":   time.time(),
        "models":      registry.list_models(),
        "audit_log":   audit_info,
    }


@app.get("/models", tags=["Platform"])
def list_models():
    """
    Returns provenance metadata for every loaded model.
    Useful for ops dashboards, compliance audits, and A/B test monitoring.
    """
    return {
        "models": registry.list_models(),
        "timestamp": time.time(),
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _path_to_domain(path: str) -> Optional[str]:
    """Map a URL path to a registry domain name, or None for non-prediction paths."""
    if path.startswith("/hiring"):
        return "hiring"
    if path.startswith("/loan"):
        return "loan"
    if path.startswith("/social"):
        return "social"
    return None


def _sanitise(payload: Optional[dict]) -> Optional[dict]:
    """
    Remove sensitive attribute keys from the payload before audit logging.
    Returns None if payload is None.
    """
    if payload is None:
        return None
    return {k: v for k, v in payload.items() if k not in _SENSITIVE_KEYS}
