"""
utils/pii.py

PII (Personally Identifiable Information) Detection and Masking Engine
======================================================================

Purpose
-------
Automatically detect and mask / hash PII from any Python object before it
is written to a log file, an audit trail, or a database record.  The masker
is applied at every log write point in utils/logger.py and at every audit
record write in log_correlation_event().

Design principles
-----------------
1.  Key-name based masking
    Any dict key whose *name* matches a known PII field is masked regardless
    of its value.  This catches ``{"email": "alice@example.com"}`` even when
    the value format doesn't look like an email.

2.  Value-pattern based scrubbing
    Free-text strings are scanned with compiled regexes that detect common
    PII patterns (email addresses, phone numbers, credit-card-like digit
    sequences, national ID formats, IP addresses, names after keywords like
    "applicant:" or "user:").  Matched spans are replaced inline.

3.  Deep traversal
    The masker walks the full object graph: dicts (keys + values), lists,
    tuples, sets, and scalar strings.  Non-string scalars (int, float, bool,
    None) are passed through unchanged — they cannot contain PII as defined
    here.

4.  Deterministic pseudonymisation (SHA-256 prefix)
    When a field is masked by *key name*, the replacement is NOT a blank
    string.  Instead it is a short pseudonym: ``<MASKED:sha256_prefix_8>``
    where the 8-character hex prefix is a deterministic SHA-256 of the
    original value.  This lets engineers:
      - Correlate "the same masked value appeared in two records" without
        recovering the original.
      - Prove in an audit that the masking is consistent across time.

5.  Regex scrubbing produces ``<REDACTED:TYPE>``
    Free-text pattern replacements use a type label, e.g. ``<REDACTED:EMAIL>``
    so log readers know *what* was removed without seeing the data itself.

6.  Allowlist for known-safe keys
    Numeric feature names used by models (``technical_score``, ``credit_score``,
    etc.) are explicitly whitelisted so that masking never silently zeros out
    model inputs.

7.  Configurable via environment variables
    PII_HASH_SALT       — salt prepended to values before hashing (default: "quantum")
    PII_MASK_ENABLED    — set to "false" to disable masking in dev/test (default: "true")

Public interface
----------------
PIIMasker (singleton ``pii_masker``)
  .mask(obj)         → same-type object with PII replaced
  .mask_str(s)       → string with PII patterns redacted
  .is_pii_key(key)   → bool
  .pseudonymise(val) → "<MASKED:xxxxxxxx>"

mask(obj)            → module-level convenience alias for pii_masker.mask(obj)
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

# ─── Configuration ────────────────────────────────────────────────────────────

_SALT:        str  = os.getenv("PII_HASH_SALT",    "quantum")
_ENABLED:     bool = os.getenv("PII_MASK_ENABLED", "true").lower() != "false"


# ─── PII field-name registry ──────────────────────────────────────────────────
# Keys whose *names* trigger masking, regardless of value content.
# Stored as a frozenset for O(1) lookup.

_PII_KEY_EXACT: frozenset[str] = frozenset({
    # Identity
    "name", "full_name", "first_name", "last_name", "surname", "given_name",
    "username", "user_name", "display_name", "handle",
    # Contact
    "email", "email_address", "phone", "phone_number", "mobile", "telephone",
    "address", "street_address", "postal_code", "zip_code", "postcode",
    # Credentials
    "password", "passwd", "secret", "token", "api_key", "api_secret",
    "access_token", "refresh_token", "auth_token", "bearer",
    "ssn", "social_security_number", "national_id", "passport_number",
    "drivers_license", "tax_id", "vat_number",
    # Financial
    "card_number", "credit_card", "debit_card", "account_number",
    "iban", "routing_number", "bank_account",
    # Biometric / health
    "dob", "date_of_birth", "birthdate", "age",
    "ip_address", "ip", "mac_address",
    # Sensitive attributes collected for fairness monitoring
    # (these are stripped at the API layer, but defence-in-depth here too)
    "sensitive_value", "sensitive_value_group",
    "gender_value", "ethnicity_value", "religion_value",
})

# Substring patterns: if any of these appear *inside* a key name, mask it.
_PII_KEY_SUBSTRINGS: tuple[str, ...] = (
    "password", "passwd", "secret", "token", "credential",
    "ssn", "dob", "birth", "passport", "license",
    "card_num", "account_num",
)

# ─── Allowlist — never mask these keys even if they match a substring ─────────
_ALLOWLIST_KEYS: frozenset[str] = frozenset({
    # Model feature names — numeric, not PII
    "years_experience", "education_level", "technical_score",
    "communication_score", "num_past_jobs", "certifications",
    "credit_score", "annual_income", "loan_amount", "loan_term_months",
    "employment_years", "existing_debt", "num_credit_lines",
    "avg_session_minutes", "posts_per_day", "topics_interacted",
    "like_rate", "share_rate", "comment_rate", "account_age_days",
    # System / audit fields
    "correlation_id", "model_version", "model_variant", "domain",
    "prediction", "confidence", "prediction_label", "timestamp",
    "timestamp_utc", "event", "path", "method", "status",
    "is_fair", "warning", "sensitive_attribute",   # attr *name* is OK; value is masked
    "bias_risk_score", "bias_risk_band", "flag_for_review",
    "shap_status", "shap_poll_url", "shap_available",
    "records_used", "sufficient_history", "message",
    # Fairness-safe fields
    "age_group",          # categorical bucket (e.g. "25-34"), not exact age
    "location",           # country/region label only — not street address
    "language",           # language code (e.g. "en")
})


# ─── Regex patterns for value-level scrubbing ─────────────────────────────────
# Each entry: (compiled_regex, replacement_label)
# Applied in order to free-text strings.

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email address
    (re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ), "EMAIL"),

    # International phone: +1 (555) 123-4567 / 07911 123456 / etc.
    (re.compile(
        r"\b(\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}\b"
    ), "PHONE"),

    # Credit / debit card (Luhn-like: 13–19 contiguous or grouped digits)
    (re.compile(
        r"\b(?:\d[ \-]?){13,18}\d\b"
    ), "CARD_NUMBER"),

    # US SSN  (123-45-6789 or 123456789)
    (re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    ), "SSN"),

    # IPv4 address
    (re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ), "IP_ADDRESS"),

    # IPv6 address (simplified)
    (re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    ), "IP_ADDRESS"),

    # IBAN  (up to 34 alphanumeric, country + 2 check digits + BBAN)
    (re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"
    ), "IBAN"),

    # Date of birth  (DD/MM/YYYY, MM-DD-YYYY, YYYY-MM-DD)
    (re.compile(
        r"\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b"
    ), "DATE"),

    # Name after "applicant:", "user:", "candidate:", "for:" (log context clues)
    (re.compile(
        r"(?:applicant|user|candidate|for|name)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        re.IGNORECASE,
    ), "PERSON_NAME"),
]


# ═════════════════════════════════════════════════════════════════════════════
# PIIMasker
# ═════════════════════════════════════════════════════════════════════════════

class PIIMasker:
    """
    Stateless PII masking engine.  All methods are pure functions with no
    side effects — safe to call from any thread or async context.
    """

    def mask(self, obj: Any) -> Any:
        """
        Deep-walk *obj* and return a copy with all PII masked.

        Type preservation
        -----------------
        dict   → dict  (same keys; PII-key values pseudonymised)
        list   → list
        tuple  → tuple
        set    → frozenset (sets are unordered; frozenset keeps semantics)
        str    → str   (pattern-scrubbed)
        int / float / bool / None → unchanged
        """
        if not _ENABLED:
            return obj
        return self._walk(obj, parent_key=None)

    def mask_str(self, s: str) -> str:
        """Apply regex scrubbing to a single string.  Key-name masking not applied."""
        if not _ENABLED or not isinstance(s, str):
            return s
        return self._scrub_string(s)

    def is_pii_key(self, key: str) -> bool:
        """Return True if *key* should have its value masked."""
        if not isinstance(key, str):
            return False
        k = key.lower().strip()
        if k in _ALLOWLIST_KEYS:
            return False
        if k in _PII_KEY_EXACT:
            return True
        return any(sub in k for sub in _PII_KEY_SUBSTRINGS)

    def pseudonymise(self, value: Any) -> str:
        """
        Replace a PII value with a deterministic pseudonym.
        Format: ``<MASKED:xxxxxxxx>`` where xxxxxxxx is the first 8 hex
        characters of SHA-256(salt + str(value)).
        """
        raw    = f"{_SALT}{value}"
        digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
        return f"<MASKED:{digest[:8]}>"

    # ── Private ───────────────────────────────────────────────────────────────

    def _walk(self, obj: Any, parent_key: str | None) -> Any:
        """Recursive deep-walker."""
        if isinstance(obj, dict):
            return self._walk_dict(obj)
        if isinstance(obj, list):
            return [self._walk(item, parent_key) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk(item, parent_key) for item in obj)
        if isinstance(obj, (set, frozenset)):
            return frozenset(self._walk(item, parent_key) for item in obj)
        if isinstance(obj, str):
            # If called on a string whose parent key is PII, the caller has
            # already replaced it; this branch handles free-text values.
            return self._scrub_string(obj)
        # int, float, bool, None, bytes — pass through
        return obj

    def _walk_dict(self, d: dict) -> dict:
        result = {}
        for key, value in d.items():
            str_key = str(key)
            if self.is_pii_key(str_key):
                # Pseudonymise by key name — don't recurse into value
                result[key] = self.pseudonymise(value) if value is not None else None
            else:
                result[key] = self._walk(value, parent_key=str_key)
        return result

    def _scrub_string(self, s: str) -> str:
        """Apply all regex patterns to a free-text string."""
        if not s:
            return s
        for pattern, label in _PII_PATTERNS:
            s = pattern.sub(f"<REDACTED:{label}>", s)
        return s


# ─── Module-level singleton and convenience alias ─────────────────────────────

pii_masker = PIIMasker()


def mask(obj: Any) -> Any:
    """Convenience alias: ``from utils.pii import mask``."""
    return pii_masker.mask(obj)
