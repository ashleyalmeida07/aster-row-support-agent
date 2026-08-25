"""
lookup_order — Order status lookup tool.

Loads orders from data/orders.json and provides a sanitized lookup function.
Uses an explicit ALLOWLIST of customer-safe fields — never a blocklist.
Handles:
  - Input normalization (lowercase, whitespace, format validation)
  - Unknown / malformed order IDs
  - Suppressing stale delivery fields for cancelled/returned orders
  - Null estimated_delivery (says unavailable, never invents a date)
  - Never exposes customer PII or internal fields
"""

import json
import re
from typing import Optional

from app.config import ORDERS_FILE


# ---------------------------------------------------------------------------
# Explicit allowlist of customer-safe fields
# ---------------------------------------------------------------------------

# These are the ONLY fields we ever return to the model.
# Everything else (customer PII, internal notes, risk scores) is stripped.
CUSTOMER_SAFE_FIELDS = {
    "order_id",
    "membership_tier",
    "items",           # We further filter item sub-fields below
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}

# Allowed sub-fields within each item
ITEM_SAFE_FIELDS = {"name", "quantity", "final_sale"}

# Statuses where delivery/shipping fields are stale and must be suppressed
STALE_DELIVERY_STATUSES = {"cancelled", "returned"}


# ---------------------------------------------------------------------------
# Load orders (cached)
# ---------------------------------------------------------------------------

_orders_cache: Optional[dict] = None
_snapshot_at: Optional[str] = None


def _load_orders() -> dict:
    """Load and cache orders.json. Returns dict mapping order_id -> order."""
    global _orders_cache, _snapshot_at
    if _orders_cache is not None:
        return _orders_cache

    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    _snapshot_at = data.get("snapshot_at")
    _orders_cache = {}
    for order in data.get("orders", []):
        oid = order.get("order_id", "").upper().strip()
        if oid:
            _orders_cache[oid] = order

    return _orders_cache


def get_snapshot_time() -> Optional[str]:
    """Return the dataset snapshot timestamp (used as 'current time' for eval)."""
    _load_orders()
    return _snapshot_at


# ---------------------------------------------------------------------------
# Input normalization and validation
# ---------------------------------------------------------------------------

ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4,}$")


def _normalize_order_id(raw_input: str) -> str:
    """
    Normalize order ID input:
    - Strip whitespace
    - Uppercase
    - Strip surrounding punctuation
    """
    cleaned = raw_input.strip().upper()
    # Remove surrounding quotes or periods
    cleaned = cleaned.strip("\"'.,;:!? ")
    return cleaned


def _validate_order_id(order_id: str) -> Optional[str]:
    """
    Validate the order ID format.
    Returns an error message if invalid, or None if valid.
    """
    if not order_id:
        return "No order ID was provided. Please provide an order ID (e.g., ORD-1007)."

    if not ORDER_ID_PATTERN.match(order_id):
        return (
            f"'{order_id}' does not look like a valid order ID. "
            f"Order IDs follow the format ORD-XXXX (e.g., ORD-1007)."
        )

    return None


# ---------------------------------------------------------------------------
# Sanitize order data (allowlist approach)
# ---------------------------------------------------------------------------

def _sanitize_order(order: dict) -> dict:
    """
    Extract only customer-safe fields from an order.
    
    - Uses an explicit ALLOWLIST — anything not listed is dropped.
    - Suppresses stale delivery fields for cancelled/returned orders.
    - Never includes customer PII or internal data.
    """
    status = order.get("status", "")
    safe = {}

    for field in CUSTOMER_SAFE_FIELDS:
        if field not in order:
            continue

        value = order[field]

        # Sanitize items: only include allowed sub-fields
        if field == "items" and isinstance(value, list):
            safe_items = []
            for item in value:
                safe_item = {k: v for k, v in item.items() if k in ITEM_SAFE_FIELDS}
                safe_items.append(safe_item)
            safe[field] = safe_items
            continue

        # Suppress stale delivery/shipping fields for cancelled/returned orders
        if status in STALE_DELIVERY_STATUSES:
            if field in ("carrier", "tracking_number", "estimated_delivery",
                         "shipped_at", "delivered_at"):
                continue

        # If estimated_delivery is null, explicitly note it
        if field == "estimated_delivery" and value is None:
            safe[field] = None
            safe["estimated_delivery_note"] = (
                "A delivery estimate is not currently available. "
                "Do not invent or calculate a date."
            )
            continue

        safe[field] = value

    return safe


# ---------------------------------------------------------------------------
# Main lookup function (the tool)
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> dict:
    """
    Look up an order by ID and return sanitized, customer-safe information.

    Args:
        order_id: The order ID to look up (e.g., "ORD-1007").

    Returns:
        A dict with:
          - "found": bool
          - "order": sanitized order data (if found)
          - "error": error message (if not found or invalid)
          - "lookup_metadata": info about the lookup operation
    """
    # Normalize input
    normalized_id = _normalize_order_id(order_id)

    # Validate format
    validation_error = _validate_order_id(normalized_id)
    if validation_error:
        return {
            "found": False,
            "order": None,
            "error": validation_error,
            "lookup_metadata": {
                "raw_input": order_id,
                "normalized_id": normalized_id,
                "action": "validation_failed",
            },
        }

    # Look up order
    orders = _load_orders()
    order = orders.get(normalized_id)

    if order is None:
        return {
            "found": False,
            "order": None,
            "error": (
                f"Order {normalized_id} was not found. "
                f"Please double-check the order ID. If the problem persists, "
                f"a human support agent can help investigate."
            ),
            "lookup_metadata": {
                "raw_input": order_id,
                "normalized_id": normalized_id,
                "action": "not_found",
            },
        }

    # Sanitize and return
    safe_order = _sanitize_order(order)

    return {
        "found": True,
        "order": safe_order,
        "error": None,
        "lookup_metadata": {
            "raw_input": order_id,
            "normalized_id": normalized_id,
            "action": "found",
            "status": safe_order.get("status", ""),
        },
    }
