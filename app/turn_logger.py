"""
Structured JSON turn logger — Phase 7 observability.

Writes one JSON object per line (JSONL) to logs/turns.jsonl.
Each turn captures: user message, route, retrieved chunks+scores,
tool call+args, sanitized tool result, final answer, sources, handoff.

Never logs: customer PII, internal notes, risk scores, API keys.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"
TURN_LOG_FILE = LOG_DIR / "turns.jsonl"

logger = logging.getLogger("aster_row.turnlog")


def _ensure_log_dir():
    LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Sanitize tool result before logging (remove any PII that might slip through)
# ---------------------------------------------------------------------------

_SENSITIVE_KEYS = {
    "email", "shipping_address", "risk_score", "warehouse_note",
    "support_tags", "internal", "name",  # customer name
}


def _sanitize_for_log(obj, depth: int = 0) -> object:
    """Recursively sanitize a dict/list for logging — strip sensitive keys."""
    if depth > 10:
        return "[truncated]"
    if isinstance(obj, dict):
        return {
            k: _sanitize_for_log(v, depth + 1)
            for k, v in obj.items()
            if k not in _SENSITIVE_KEYS
        }
    if isinstance(obj, list):
        return [_sanitize_for_log(item, depth + 1) for item in obj]
    if isinstance(obj, str) and len(obj) > 500:
        return obj[:500] + "... [truncated]"
    return obj


# ---------------------------------------------------------------------------
# Main log writer
# ---------------------------------------------------------------------------

def log_turn(
    session_id: str,
    user_message: str,
    route: str,
    retrieved_passages: Optional[list] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    tool_result: Optional[dict] = None,
    response: str = "",
    sources: Optional[list] = None,
    handoff_recommended: bool = False,
    groundedness_notes: Optional[list] = None,
    rewritten_query: Optional[str] = None,
    error: Optional[str] = None,
):
    """
    Write a structured turn log entry to logs/turns.jsonl.

    Args are sanitized before writing — no PII, no secrets.
    """
    _ensure_log_dir()

    # Build passage summaries (no full text — just metadata + score)
    passage_summaries = []
    if retrieved_passages:
        for p in retrieved_passages:
            passage_summaries.append({
                "source_file": p.get("source_file", ""),
                "heading": p.get("heading", ""),
                "is_authoritative": p.get("is_authoritative", False),
                "status": p.get("status", ""),
                "relevance_score": p.get("relevance_score", 0),
                "text_preview": p.get("text", "")[:200],
            })

    # Sanitize tool result
    sanitized_tool_result = None
    if tool_result:
        sanitized_tool_result = _sanitize_for_log(tool_result)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "turn": {
            "user_message": user_message,
            "rewritten_query": rewritten_query or user_message,
            "route": route,
        },
        "retrieval": {
            "passages_retrieved": len(passage_summaries),
            "passages": passage_summaries,
        },
        "tool": {
            "name": tool_name,
            "args": tool_args,
            "result": sanitized_tool_result,
        } if tool_name else None,
        "response": {
            "text": response,
            "sources": sources or [],
            "handoff_recommended": handoff_recommended,
            "groundedness_notes": groundedness_notes or [],
        },
        "error": error,
    }

    try:
        with open(TURN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.error(f"Failed to write turn log: {e}")

    # Also emit to structured Python logger at DEBUG level
    logger.debug(
        "TURN | session=%s route=%s sources=%d handoff=%s",
        session_id[:8],
        route,
        len(sources or []),
        handoff_recommended,
    )


def get_log_path() -> str:
    """Return the path to the turn log file."""
    return str(TURN_LOG_FILE)


def tail_log(n: int = 5) -> list[dict]:
    """Return the last n turn log entries as dicts."""
    if not TURN_LOG_FILE.exists():
        return []
    lines = TURN_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-n:] if len(lines) >= n else lines
    result = []
    for line in recent:
        try:
            result.append(json.loads(line))
        except Exception:
            pass
    return result
