from __future__ import annotations
from typing import Iterable,Mapping,Sequence
import numpy as np
PRICE_PROTOCOL_VERSION=1
LEGACY_PRICE_SIGNAL='price'
LEGACY_PRICE_SEQ_FIELD='price_seq'
LEGACY_PRICE_PRED_COLUMN='price_pred'
WHOLESALE_PRICE_SIGNAL='wholesale_price'
WHOLESALE_PRICE_SEQ_FIELD='wholesale_price_seq'
WHOLESALE_PRICE_PRED_COLUMN='wholesale_price_pred'
IMPORT_PRICE_COLUMN='import_price'
IMPORT_PRICE_SEQ_FIELD='import_price_seq'
IMPORT_PRICE_PRED_COLUMN='import_price_pred'
IMPORT_PRICE_MARKUP_KEY='import_price_markup_eur_per_kwh'
DEFAULT_IMPORT_PRICE_MARKUP_EUR_PER_KWH=.2
LEGACY_PRICE_PROTOCOL_KEYS=frozenset({LEGACY_PRICE_SIGNAL,LEGACY_PRICE_SEQ_FIELD,LEGACY_PRICE_PRED_COLUMN,'import_price_adder_eur_per_kwh'})
class PriceProtocolError(ValueError):0
class LegacyPriceSchemaError(PriceProtocolError):0
class PriceProtocolMismatchError(PriceProtocolError):0
def normalize_internal_signal_name(signal_name:str)->str:
	normalized=str(signal_name).strip().lower()
	if normalized==LEGACY_PRICE_SIGNAL:raise LegacyPriceSchemaError(f"Legacy internal signal name 'price' is no longer supported. Use '{WHOLESALE_PRICE_SIGNAL}' at all internal interfaces.")
	return normalized
def derive_import_price(wholesale_price:float|np.ndarray,*,markup_eur_per_kwh:float)->float|np.ndarray:return np.asarray(wholesale_price,dtype=np.float32)+np.float32(markup_eur_per_kwh)
def derive_import_price_seq(wholesale_price_seq:Sequence[float]|np.ndarray,*,markup_eur_per_kwh:float)->np.ndarray:return np.asarray(wholesale_price_seq,dtype=np.float32).reshape(-1)+np.float32(markup_eur_per_kwh)
def get_import_price_markup(cfg_or_reward:object|None,*,default:float=DEFAULT_IMPORT_PRICE_MARKUP_EUR_PER_KWH)->float:
	if cfg_or_reward is None:return float(default)
	reward_cfg=getattr(cfg_or_reward,'reward',cfg_or_reward);return float(getattr(reward_cfg,IMPORT_PRICE_MARKUP_KEY,default))
def detect_legacy_price_schema_keys(keys:Iterable[object])->list[str]:normalized={str(key)for key in keys};return sorted(str(key)for key in LEGACY_PRICE_PROTOCOL_KEYS if str(key)in normalized)
def assert_no_legacy_price_schema(keys:Iterable[object],*,context:str)->None:
	legacy_keys=detect_legacy_price_schema_keys(keys)
	if legacy_keys:raise LegacyPriceSchemaError(f"{context} uses legacy price schema keys {legacy_keys}. Use '{WHOLESALE_PRICE_SIGNAL}', '{WHOLESALE_PRICE_SEQ_FIELD}', '{WHOLESALE_PRICE_PRED_COLUMN}', '{IMPORT_PRICE_COLUMN}', '{IMPORT_PRICE_SEQ_FIELD}', '{IMPORT_PRICE_PRED_COLUMN}', and '{IMPORT_PRICE_MARKUP_KEY}' instead.")
