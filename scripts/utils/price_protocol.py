from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

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
DEFAULT_IMPORT_PRICE_MARKUP_EUR_PER_KWH = 0.2

LEGACY_PRICE_PROTOCOL_KEYS = frozenset(
    {
        LEGACY_PRICE_SIGNAL,
        LEGACY_PRICE_SEQ_FIELD,
        LEGACY_PRICE_PRED_COLUMN,
        "import_price_adder_eur_per_kwh",
    }
)


class PriceProtocolError(ValueError):
    pass


class LegacyPriceSchemaError(PriceProtocolError):
    pass


class PriceProtocolMismatchError(PriceProtocolError):
    pass


def normalize_internal_signal_name(signal_name: str) -> str:
    normalized = str(signal_name).strip().lower()
    if normalized == LEGACY_PRICE_SIGNAL:
        raise LegacyPriceSchemaError(
            "Legacy internal signal name 'price' is no longer supported. "
            f"Use '{WHOLESALE_PRICE_SIGNAL}' at all internal interfaces."
        )
    return normalized


def derive_import_price(
    wholesale_price: float | np.ndarray, *, markup_eur_per_kwh: float
) -> float | np.ndarray:
    return np.asarray(wholesale_price, dtype=np.float32) + np.float32(markup_eur_per_kwh)


def derive_import_price_seq(
    wholesale_price_seq: Sequence[float] | np.ndarray, *, markup_eur_per_kwh: float
) -> np.ndarray:
    return np.asarray(wholesale_price_seq, dtype=np.float32).reshape(-1) + np.float32(markup_eur_per_kwh)


def derive_wholesale_price(
    import_price: float | np.ndarray, *, markup_eur_per_kwh: float
) -> float | np.ndarray:
    return np.asarray(import_price, dtype=np.float32) - np.float32(markup_eur_per_kwh)


def derive_wholesale_price_seq(
    import_price_seq: Sequence[float] | np.ndarray, *, markup_eur_per_kwh: float
) -> np.ndarray:
    return np.asarray(import_price_seq, dtype=np.float32).reshape(-1) - np.float32(markup_eur_per_kwh)


def get_import_price_markup(
    cfg_or_reward: object | None, *, default: float = DEFAULT_IMPORT_PRICE_MARKUP_EUR_PER_KWH
) -> float:
    if cfg_or_reward is None:
        return float(default)
    reward_cfg = getattr(cfg_or_reward, "reward", cfg_or_reward)
    if isinstance(reward_cfg, Mapping):
        return float(reward_cfg.get(IMPORT_PRICE_MARKUP_KEY, default))
    return float(getattr(reward_cfg, IMPORT_PRICE_MARKUP_KEY, default))


def require_import_price_markup(cfg_or_reward: object | None, *, context: str) -> float:
    if cfg_or_reward is None:
        raise PriceProtocolMismatchError(
            f"{context} is missing '{IMPORT_PRICE_MARKUP_KEY}', so import prices cannot be validated."
        )
    reward_cfg = getattr(cfg_or_reward, "reward", cfg_or_reward)
    if isinstance(reward_cfg, Mapping):
        if IMPORT_PRICE_MARKUP_KEY not in reward_cfg:
            raise PriceProtocolMismatchError(
                f"{context} is missing '{IMPORT_PRICE_MARKUP_KEY}', so import prices cannot be validated."
            )
        markup = reward_cfg.get(IMPORT_PRICE_MARKUP_KEY)
        if markup is None:
            raise PriceProtocolMismatchError(
                f"{context} has '{IMPORT_PRICE_MARKUP_KEY}=None', so import prices cannot be validated."
            )
        return float(markup)
    if not hasattr(reward_cfg, IMPORT_PRICE_MARKUP_KEY):
        raise PriceProtocolMismatchError(
            f"{context} is missing '{IMPORT_PRICE_MARKUP_KEY}', so import prices cannot be validated."
        )
    markup = getattr(reward_cfg, IMPORT_PRICE_MARKUP_KEY)
    if markup is None:
        raise PriceProtocolMismatchError(
            f"{context} has '{IMPORT_PRICE_MARKUP_KEY}=None', so import prices cannot be validated."
        )
    return float(markup)


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


def _as_price_array(values: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _canonicalize_price_pair(
    *,
    wholesale_values: Sequence[float] | np.ndarray | None,
    import_values: Sequence[float] | np.ndarray | None,
    markup_eur_per_kwh: float | None,
    context: str,
    pair_label: str,
    allow_missing: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    wholesale_array = _as_price_array(wholesale_values)
    import_array = _as_price_array(import_values)
    if wholesale_array is None and import_array is None:
        if allow_missing:
            return None, None
        raise PriceProtocolMismatchError(
            f"{context} is missing both wholesale and import {pair_label} price values."
        )
    if markup_eur_per_kwh is None:
        if wholesale_array is None or import_array is None:
            raise PriceProtocolMismatchError(
                f"{context} needs '{IMPORT_PRICE_MARKUP_KEY}' to derive missing {pair_label} import prices."
            )
        return wholesale_array.astype(np.float32), import_array.astype(np.float32)
    if wholesale_array is None:
        wholesale_array = derive_wholesale_price_seq(import_array, markup_eur_per_kwh=markup_eur_per_kwh)
    if import_array is None:
        import_array = derive_import_price_seq(wholesale_array, markup_eur_per_kwh=markup_eur_per_kwh)
    expected_import = derive_import_price_seq(wholesale_array, markup_eur_per_kwh=markup_eur_per_kwh)
    if expected_import.shape != import_array.shape or not np.allclose(
        expected_import,
        import_array,
        atol=1e-6,
        rtol=1e-6,
    ):
        max_abs_err = (
            float(np.max(np.abs(expected_import - import_array)))
            if expected_import.shape == import_array.shape and expected_import.size
            else float("nan")
        )
        raise PriceProtocolMismatchError(
            f"{context} has inconsistent {pair_label} import prices. "
            f"Expected import = wholesale + {markup_eur_per_kwh:.6f}, "
            f"max_abs_err={max_abs_err!r}."
        )
    return wholesale_array.astype(np.float32), import_array.astype(np.float32)


def canonicalize_step_price_frame(
    step_df: pd.DataFrame,
    *,
    markup_eur_per_kwh: float | None,
    context: str,
    require_actual: bool = True,
    require_prediction: bool = False,
) -> pd.DataFrame:
    if step_df.empty:
        return step_df.copy()
    result = step_df.copy()
    wholesale_actual, import_actual = _canonicalize_price_pair(
        wholesale_values=result[WHOLESALE_PRICE_SIGNAL].to_numpy(dtype=np.float32)
        if WHOLESALE_PRICE_SIGNAL in result.columns
        else None,
        import_values=result[IMPORT_PRICE_COLUMN].to_numpy(dtype=np.float32)
        if IMPORT_PRICE_COLUMN in result.columns
        else None,
        markup_eur_per_kwh=markup_eur_per_kwh,
        context=context,
        pair_label="actual",
        allow_missing=not require_actual,
    )
    if wholesale_actual is not None:
        result[WHOLESALE_PRICE_SIGNAL] = wholesale_actual
    if import_actual is not None:
        result[IMPORT_PRICE_COLUMN] = import_actual
    wholesale_pred, import_pred = _canonicalize_price_pair(
        wholesale_values=result[WHOLESALE_PRICE_PRED_COLUMN].to_numpy(dtype=np.float32)
        if WHOLESALE_PRICE_PRED_COLUMN in result.columns
        else None,
        import_values=result[IMPORT_PRICE_PRED_COLUMN].to_numpy(dtype=np.float32)
        if IMPORT_PRICE_PRED_COLUMN in result.columns
        else None,
        markup_eur_per_kwh=markup_eur_per_kwh,
        context=context,
        pair_label="predicted",
        allow_missing=not require_prediction,
    )
    if wholesale_pred is not None:
        result[WHOLESALE_PRICE_PRED_COLUMN] = wholesale_pred
    if import_pred is not None:
        result[IMPORT_PRICE_PRED_COLUMN] = import_pred
    return result
