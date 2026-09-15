"""Shared helpers: logging, text normalization, month parsing, country labels."""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from typing import Any

import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\s\-_/&+]")


def get_logger(name: str = "pricing_engine") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def normalize_text(value: Any) -> str:
    """Lowercase, strip accents where safe, collapse whitespace.

    Accent stripping is conservative: Vietnamese/Thai tokens are kept when
    Unicode category is a letter. We only remove punctuation noise.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    text = unicodedata.normalize("NFKC", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip().lower()


def title_case_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _WHITESPACE_RE.sub(" ", str(value).strip())


def map_alias(value: Any, aliases: dict[str, str], *, default: str | None = None) -> str:
    raw = title_case_label(value)
    if not raw:
        return default or ""
    key = raw.lower()
    if key in aliases:
        return aliases[key]
    return raw


def parse_month(value: Any) -> pd.Period | pd.NaT:
    """Parse heterogeneous month fields into a monthly Period.

    Accepts 2026-03, 2026/03, 2026-03-15, 202603, and pandas timestamps.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Period(value, freq="M")

    text = str(value).strip()
    if not text:
        return pd.NaT

    formats = (
        ("%Y-%m-%d", 10),
        ("%Y/%m/%d", 10),
        ("%Y-%m", 7),
        ("%Y/%m", 7),
        ("%Y%m", 6),
    )
    for fmt, width in formats:
        chunk = text[:width]
        try:
            return pd.Period(datetime.strptime(chunk, fmt), freq="M")
        except ValueError:
            continue

    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Period(ts, freq="M")


def month_to_str(value: Any) -> str:
    period = parse_month(value)
    if period is pd.NaT or pd.isna(period):
        return ""
    return str(period)


def infer_marketplace_from_url(url: Any) -> str:
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return ""
    text = str(url).lower()
    if "shopee" in text:
        return "Shopee"
    if "lazada" in text:
        return "Lazada"
    return ""


def safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.replace(",", "").replace("%", "").strip()
        if text == "":
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return float(number)
