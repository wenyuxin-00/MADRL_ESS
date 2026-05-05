from __future__ import annotations

from typing import Any
import numpy as np

IMPORT_PRICE_MARKUP_KEY = "import_price_markup_eur_per_kwh"
WHOLESALE_PRICE_SEQ_FIELD = "wholesale_price_seq"


def derive_import_price_seq(wholesale_price_seq: np.ndarray, *, markup_eur_per_kwh: float = 0.0) -> np.ndarray:
    return (np.asarray(wholesale_price_seq, dtype=np.float32).reshape(-1) + np.float32(markup_eur_per_kwh)).astype(np.float32)


def get_import_price_markup(source: Any) -> float:
    if isinstance(source, dict):
        return float(source[IMPORT_PRICE_MARKUP_KEY])
    reward = getattr(source, "reward", source)
    return float(getattr(reward, IMPORT_PRICE_MARKUP_KEY))


def require_import_price_markup(meta: dict[str, Any], *, context: str) -> float:
    if IMPORT_PRICE_MARKUP_KEY not in meta:
        raise KeyError(f"{context} expected {IMPORT_PRICE_MARKUP_KEY}.")
    return float(meta[IMPORT_PRICE_MARKUP_KEY])
