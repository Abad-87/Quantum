"""
Quantum - Unbiased AI Decision Platform
Main entry point for the FastAPI application.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from hiring.router import router as hiring_router
from loan.router import router as loan_router
from social.router import router as social_router
from utils.logger import setup_logger

# Setup application logger
logger = setup_logger("main")

# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Quantum – Unbiased AI Decision Platform",
    description=(
        "A fairness-aware AI backend for Job Hiring, Loan Approval, "
        "and Social Media Recommendation. Built for hackathon."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc UI at /redoc
)

# ---------------------------------------------------------------------------
# CORS – allow React frontend (any origin during dev; restrict in production)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request timing middleware (logs how long each request takes)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{duration}ms]")
    return response

# ---------------------------------------------------------------------------
# Global exception handler – catches unexpected errors gracefully
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# ---------------------------------------------------------------------------
# Register domain routers
# ---------------------------------------------------------------------------
app.include_router(hiring_router, prefix="/hiring", tags=["Hiring"])
app.include_router(loan_router,   prefix="/loan",   tags=["Loan"])
app.include_router(social_router, prefix="/social", tags=["Social"])

# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "platform": "Quantum – Unbiased AI Decision Platform",
        "endpoints": [
            "POST /hiring/predict",
            "POST /loan/predict",
            "POST /social/recommend",
            "GET  /docs",
        ],
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "timestamp": time.time()}
