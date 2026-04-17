"""
Quantum — Unbiased AI Decision Platform
main.py  —  Phase 4: Security & Privacy

Phase 4 additions
-----------------
1. Custom Pydantic validation error handler — returns ValidationErrorResponse
   (never raw input values or stack traces).
2. Request security middleware — body-size limit, null-byte scan,
   Content-Type enforcement on prediction routes, security response headers.
3. PII-masked logging via utils/pii.py applied at every log write point.
4. utils/validation.py — all schemas live there; routers import from it.
5. Version 4.0.0.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from hiring.router    import router as hiring_router
from loan.router      import router as loan_router
from social.router    import router as social_router
from utils.shap_cache import router as shap_router
from utils.logger     import setup_logger, log_correlation_event
from utils.model_registry import registry
from utils.database   import ensure_indexes
from utils.validation import (
    ValidationErrorResponse, ValidationErrorDetail, SecurityErrorResponse
)

import hiring.model_loader as hiring_loader
import loan.model_loader   as loan_loader
import social.model_loader as social_loader

logger = setup_logger("main")

MAX_BODY_BYTES: int = int(os.getenv("MAX_BODY_BYTES", str(64 * 1024)))

_SENSITIVE_KEYS = frozenset({
    "gender", "religion", "ethnicity", "race",
    "age_group", "location", "language", "disability",
})
_PREDICTION_PATHS = frozenset({"/hiring/predict", "/loan/predict", "/social/recommend"})


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Quantum startup (Phase 4) ===")
    try:
        hiring_loader.preload()
        loan_loader.preload()
        social_loader.preload()
    except FileNotFoundError as exc:
        logger.error(f"FATAL: {exc}  Run python create_dummy_models.py")
        raise
    await ensure_indexes()
    logger.info("=== Ready ===")
    yield
    logger.info("=== Shutdown ===")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Quantum – Unbiased AI Decision Platform",
    description = (
        "Fairness-aware AI backend — Phase 4 hardened: strict Pydantic v2 "
        "validation with injection guards, PII-masked append-only audit logs, "
        "body-size limits, and structured error responses."
    ),
    version     = "4.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ─── Custom error handlers ────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Replace FastAPI's default 422 body with ValidationErrorResponse.
    Never exposes raw input values — only field name + human-readable message.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    details: List[ValidationErrorDetail] = []
    for error in exc.errors():
        field = ".".join(
            str(part) for part in error.get("loc", ()) if part != "body"
        )
        details.append(ValidationErrorDetail(
            field   = field or "unknown",
            message = error.get("msg", "Invalid value"),
            input   = None,   # never echo raw input back
        ))
    logger.warning(
        f"[{correlation_id}] Validation error on {request.url.path}: "
        f"{len(details)} field(s) failed"
    )
    return JSONResponse(
        status_code = 422,
        content     = ValidationErrorResponse(
            error=          "Validation failed",
            correlation_id= correlation_id,
            details=        details,
        ).model_dump(),
    )


@app.exception_handler(ValidationError)
async def pydantic_handler(request: Request, exc: ValidationError):
    correlation_id = getattr(request.state, "correlation_id", None)
    return JSONResponse(
        status_code = 422,
        content     = ValidationErrorResponse(
            error=          "Data validation error",
            correlation_id= correlation_id,
            details=        [ValidationErrorDetail(field="unknown", message=str(exc))],
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.error(f"[{correlation_id}] Unhandled on {request.url.path}: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":          "Internal server error",
            "correlation_id": correlation_id,
            # No 'detail' key — never leak exception messages to clients
        },
    )


# ─── Security middleware (outermost — registered last) ────────────────────────

@app.middleware("http")
async def request_security_middleware(request: Request, call_next):
    """
    Guards (short-circuit on first failure):
    1. Content-Length pre-check → 413 if over MAX_BODY_BYTES.
    2. Body size + null-byte scan.
    3. Content-Type enforcement on prediction endpoints → 415.
    Then adds security response headers to every response.
    """
    path   = request.url.path
    method = request.method

    # 1. Content-Length header pre-check
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_BODY_BYTES:
                logger.warning(f"[security] Oversized Content-Length={cl} path={path}")
                return JSONResponse(
                    status_code = 413,
                    content = SecurityErrorResponse(
                        reason=f"Request body exceeds {MAX_BODY_BYTES} bytes."
                    ).model_dump(),
                )
        except ValueError:
            pass

    if method in ("POST", "PUT", "PATCH"):
        body_bytes = await request.body()

        # 2a. Actual body size
        if len(body_bytes) > MAX_BODY_BYTES:
            logger.warning(f"[security] Oversized body {len(body_bytes)}B path={path}")
            return JSONResponse(
                status_code = 413,
                content = SecurityErrorResponse(
                    reason=f"Request body exceeds {MAX_BODY_BYTES} bytes."
                ).model_dump(),
            )

        # 2b. Null-byte injection
        if b"\x00" in body_bytes:
            logger.warning(f"[security] Null-byte injection attempt path={path}")
            return JSONResponse(
                status_code = 400,
                content = SecurityErrorResponse(
                    reason="Request body contains disallowed characters."
                ).model_dump(),
            )

        # 3. Content-Type on prediction routes
        if path in _PREDICTION_PATHS:
            ct = request.headers.get("content-type", "")
            if "application/json" not in ct.lower():
                logger.warning(f"[security] Non-JSON Content-Type='{ct}' path={path}")
                return JSONResponse(
                    status_code = 415,
                    content = SecurityErrorResponse(
                        reason="Content-Type must be 'application/json'."
                    ).model_dump(),
                )

    response = await call_next(request)

    # Security response headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"

    return response


# ─── Correlation middleware (inner layer) ─────────────────────────────────────

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
    method            = request.method
    domain            = _path_to_domain(path)
    model_meta        = registry.get_metadata(domain) if domain else {}
    sanitised_payload = _sanitise(raw_payload)

    if path not in ("/health", "/", "/docs", "/redoc", "/openapi.json"):
        log_correlation_event(
            correlation_id = correlation_id,
            event          = "request_received",
            path           = path,
            method         = method,
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
        f"{method} {path} → {response.status_code} "
        f"[{elapsed_ms}ms] corr={correlation_id[:8]}…"
    )
    return response


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
        "version":  "4.0.0",
        "security": {
            "pii_masking":      os.getenv("PII_MASK_ENABLED", "true"),
            "input_validation": "strict Pydantic v2 + injection guards",
            "body_size_limit":  f"{MAX_BODY_BYTES // 1024} KB",
            "response_headers": "nosniff, deny-framing, referrer-policy",
        },
        "endpoints": [
            "POST /hiring/predict",
            "POST /loan/predict",
            "POST /social/recommend",
            "GET  /shap/{correlation_id}",
            "WS   /shap/ws/{correlation_id}",
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
        "version":   "4.0.0",
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
        "security": {
            "pii_masking":    os.getenv("PII_MASK_ENABLED", "true"),
            "max_body_bytes": MAX_BODY_BYTES,
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
