"""Canonical price protocol helpers shared across data, env, MPC, and reports."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

PRICE_PROTOCOL_VERSION = 1

LEGACY_PRICE_SIGNAL = "price"
LEGACY_PRICE_SEQ_FIELD = "price_seq"
LEGACY_PRICE_PRED_COLUMN = "price_pred"

WHOLESALE_PRICE_SIGNAL = "wholesale_price"
WHOLESALE_PRICE_SEQ_FIELD = "wholesale_price_seq"
WHOLESALE_PRICE_PRED_COLUMN = "wholesale_price_pred"

IMPORT_PRICE_COLUMN = "import_price"
IMPORT_PRICE_SEQ_FIELD = "import_price_seq"
IMPORT_PRICE_PRED_COLUMN = "import_price_pred"

IMPORT_PRICE_MARKUP_KEY = "import_price_markup_eur_per_kwh"
DEFAULT_IMPORT_PRICE_MARKUP_EUR_PER_KWH = 0.20

LEGACY_PRICE_PROTOCOL_KEYS = frozenset(
    {
        LEGACY_PRICE_SIGNAL,
        LEGACY_PRICE_SEQ_FIELD,
        LEGACY_PRICE_PRED_COLUMN,
        "import_price_adder_eur_per_kwh",
    }
)


class PriceProtocolError(ValueError):
    """Base class for price-protocol failures."""


class LegacyPriceSchemaError(PriceProtocolError):
    """Raised when old price schema keys are encountered on internal interfaces."""


class PriceProtocolMismatchError(PriceProtocolError):
    """Raised when a cached payload/artifact uses a different price protocol version."""


def normalize_internal_signal_name(signal_name: str) -> str:
    normalized = str(signal_name).strip().lower()
    if normalized == LEGACY_PRICE_SIGNAL:
        raise LegacyPriceSchemaError(
            "Legacy internal signal name 'price' is no longer supported. "
            f"Use '{WHOLESALE_PRICE_SIGNAL}' at all internal interfaces."
        )
    return normalized


def derive_import_price(
    wholesale_price: float | np.ndarray,
    *,
    markup_eur_per_kwh: float,
) -> float | np.ndarray:
    return np.asarray(wholesale_price, dtype=np.float32) + np.float32(markup_eur_per_kwh)


def derive_import_price_seq(
    wholesale_price_seq: Sequence[float] | np.ndarray,
    *,
    markup_eur_per_kwh: float,
) -> np.ndarray:
    return np.asarray(wholesale_price_seq, dtype=np.float32).reshape(-1) + np.float32(markup_eur_per_kwh)


def get_import_price_markup(
    cfg_or_reward: object | None,
    *,
    default: float = DEFAULT_IMPORT_PRICE_MARKUP_EUR_PER_KWH,
) -> float:
    if cfg_or_reward is None:
        return float(default)
    reward_cfg = getattr(cfg_or_reward, "reward", cfg_or_reward)
    return float(getattr(reward_cfg, IMPORT_PRICE_MARKUP_KEY, default))


def detect_legacy_price_schema_keys(keys: Iterable[object]) -> list[str]:
    normalized = {str(key) for key in keys}
    return sorted(str(key) for key in LEGACY_PRICE_PROTOCOL_KEYS if str(key) in normalized)


def assert_no_legacy_price_schema(keys: Iterable[object], *, context: str) -> None:
    legacy_keys = detect_legacy_price_schema_keys(keys)
    if legacy_keys:
        raise LegacyPriceSchemaError(
            f"{context} uses legacy price schema keys {legacy_keys}. "
            f"Use '{WHOLESALE_PRICE_SIGNAL}', '{WHOLESALE_PRICE_SEQ_FIELD}', "
            f"'{WHOLESALE_PRICE_PRED_COLUMN}', '{IMPORT_PRICE_COLUMN}', "
            f"'{IMPORT_PRICE_SEQ_FIELD}', '{IMPORT_PRICE_PRED_COLUMN}', and "
            f"'{IMPORT_PRICE_MARKUP_KEY}' instead."
        )


def assert_price_protocol_version(
    payload: Mapping[str, object],
    *,
    context: str,
    field_name: str = "price_protocol_version",
) -> None:
    actual_version = payload.get(field_name)
    if int(actual_version) != int(PRICE_PROTOCOL_VERSION):
        raise PriceProtocolMismatchError(
            f"{context} uses unsupported price protocol version: "
            f"expected={PRICE_PROTOCOL_VERSION}, actual={actual_version!r}."
        )
