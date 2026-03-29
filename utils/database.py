"""
utils/database.py

Database integration using MongoDB (via Motor for async).
Falls back to a local JSON file if MongoDB is not configured.

Set the MONGO_URI environment variable to connect to MongoDB Atlas.
If not set, predictions are appended to predictions.json (good for local dev).
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("database")

# ─── JSON fallback path ──────────────────────────────────────────────────────
JSON_LOG_PATH = Path("predictions.json")

# ─── Try to import Motor (async MongoDB driver) ──────────────────────────────
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    logger.warning("Motor not installed — using JSON fallback for storage.")

MONGO_URI = os.getenv("MONGO_URI", "")
_client = None
_db = None


def get_database():
    """
    Returns the MongoDB database instance.
    Call once at startup; reuse the client across requests.
    """
    global _client, _db

    if not MONGO_URI or not MOTOR_AVAILABLE:
        return None  # Use JSON fallback

    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
        _db = _client["quantum"]
        logger.info("Connected to MongoDB Atlas.")

    return _db


async def save_prediction(record: dict):
    """
    Persists a prediction record to MongoDB or JSON fallback.

    Record structure:
    {
        domain, input, prediction, prediction_label,
        explanation, fairness, timestamp
    }
    """
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    db = get_database()

    if db is not None:
        # ── MongoDB path ────────────────────────────────────────────────────
        try:
            collection = db["predictions"]
            await collection.insert_one(record)
            logger.debug(f"Saved prediction to MongoDB [{record['domain']}]")
        except Exception as e:
            logger.error(f"MongoDB write failed: {e}. Falling back to JSON.")
            _append_to_json(record)
    else:
        # ── JSON fallback path ──────────────────────────────────────────────
        _append_to_json(record)


def _append_to_json(record: dict):
    """Appends a record to the local JSON log file."""
    records = []

    if JSON_LOG_PATH.exists():
        try:
            with open(JSON_LOG_PATH, "r") as f:
                records = json.load(f)
        except json.JSONDecodeError:
            records = []

    records.append(record)

    with open(JSON_LOG_PATH, "w") as f:
        json.dump(records, f, indent=2)

    logger.debug(f"Saved prediction to {JSON_LOG_PATH}")


async def get_recent_predictions(domain: str, limit: int = 100) -> list:
    """
    Retrieves recent predictions for a domain.
    Used by the fairness monitoring system.
    """
    db = get_database()

    if db is not None:
        try:
            cursor = db["predictions"].find(
                {"domain": domain},
                sort=[("timestamp", -1)],
                limit=limit,
            )
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"MongoDB read failed: {e}")
            return []
    else:
        # JSON fallback
        if not JSON_LOG_PATH.exists():
            return []
        try:
            with open(JSON_LOG_PATH, "r") as f:
                all_records = json.load(f)
            domain_records = [r for r in all_records if r.get("domain") == domain]
            return domain_records[-limit:]
        except Exception:
            return []
