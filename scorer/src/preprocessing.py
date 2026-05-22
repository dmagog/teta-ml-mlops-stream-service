"""Этап пайплайна: препроцессинг транзакций формата test.csv (sparkov).

Те же признаки используются при обучении модели (model/train.py импортирует
этот модуль), поэтому train и serve не расходятся. Может запускаться как скрипт.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_REQUIRED_COLUMNS = [
    "trans_num", "trans_date_trans_time", "cc_num", "merchant", "category",
    "amt", "gender", "city", "state", "zip", "lat", "long", "city_pop",
    "job", "dob", "unix_time", "merch_lat", "merch_long",
]

NUMERIC_FEATURES = [
    "hour", "dayofweek", "month", "amt", "amt_log1p",
    "distance_km", "age", "city_pop", "city_pop_log1p",
]

CATEGORICAL_FEATURES = ["category", "merchant", "state", "gender", "job"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    ts = pd.to_datetime(raw["trans_date_trans_time"], errors="coerce")
    dob = pd.to_datetime(raw["dob"], errors="coerce")
    out["hour"] = ts.dt.hour.fillna(0).astype(int)
    out["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    out["month"] = ts.dt.month.fillna(0).astype(int)

    amt = pd.to_numeric(raw["amt"], errors="coerce").fillna(0.0)
    out["amt"] = amt
    out["amt_log1p"] = np.log1p(amt.clip(lower=0))

    out["distance_km"] = _haversine_km(
        pd.to_numeric(raw["lat"], errors="coerce").fillna(0.0),
        pd.to_numeric(raw["long"], errors="coerce").fillna(0.0),
        pd.to_numeric(raw["merch_lat"], errors="coerce").fillna(0.0),
        pd.to_numeric(raw["merch_long"], errors="coerce").fillna(0.0),
    )

    age = (ts - dob).dt.days / 365.25
    out["age"] = age.fillna(age.median() if age.notna().any() else 40.0)

    city_pop = pd.to_numeric(raw["city_pop"], errors="coerce").fillna(0.0)
    out["city_pop"] = city_pop
    out["city_pop_log1p"] = np.log1p(city_pop.clip(lower=0))

    for col in CATEGORICAL_FEATURES:
        out[col] = raw[col].fillna("__missing__").astype(str)

    return out


def run_preproc(raw: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Признаки в порядке колонок модели; вход реиндексируется к RAW_REQUIRED_COLUMNS
    (лишнее отбрасывается, пропуски -> NaN), поэтому препроцессинг самодостаточен."""
    feature_columns = (config or {}).get("feature_columns", FEATURE_COLUMNS)
    raw = raw.reindex(columns=RAW_REQUIRED_COLUMNS)
    features = build_features(raw)[feature_columns].copy()
    logger.info("Препроцессинг: %d строк, %d признаков.", len(features), features.shape[1])
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Препроцессинг сырых транзакций.")
    parser.add_argument("input_path", type=Path, help="CSV формата test.csv")
    parser.add_argument("--out", type=Path, default=None, help="Куда сохранить признаки (parquet)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    features = run_preproc(pd.read_csv(args.input_path))
    if args.out is not None:
        features.to_parquet(args.out)
    else:
        print(features.head().to_string())


if __name__ == "__main__":
    sys.exit(main())
