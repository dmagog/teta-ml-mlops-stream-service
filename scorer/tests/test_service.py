"""Тесты сервиса: контракт препроцессинга и инференса.

Главная цель — поймать train/serve skew: набор и порядок признаков сервиса
должны совпадать с тем, на чём обучена модель (feature_config.json), а скор
модели — быть валидной вероятностью.

Запуск:  cd scorer && python -m pytest -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_feature_config
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_REQUIRED_COLUMNS,
    run_preproc,
)
from src.scorer import predict_scores, score_to_flag


def _sample_raw(n: int = 5) -> pd.DataFrame:
    base = {
        "trans_date_trans_time": "2025-05-10 23:30:00",
        "cc_num": 1234567890123456,
        "merchant": "Kuhn LLC",
        "category": "misc_net",
        "amt": 250.0,
        "gender": "M",
        "city": "City_NY",
        "state": "NY",
        "zip": 10001,
        "lat": 40.7,
        "long": -74.0,
        "city_pop": 1_000_000,
        "job": "Engineer",
        "dob": "1985-06-15",
        "unix_time": 1_746_900_000,
        "merch_lat": 45.0,
        "merch_long": -79.0,
    }
    return pd.DataFrame([dict(base, trans_num=f"x{i}") for i in range(n)])


def test_preprocessing_column_contract():
    feats = run_preproc(_sample_raw(4))
    # ровно те же колонки и в том же порядке, что ждёт модель
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats.shape[0] == 4


def test_numeric_features_no_nan_and_numeric():
    feats = run_preproc(_sample_raw(4))
    num = feats[NUMERIC_FEATURES]
    assert not num.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(num[c]) for c in NUMERIC_FEATURES)


def test_categorical_features_are_strings():
    feats = run_preproc(_sample_raw(4))
    for col in CATEGORICAL_FEATURES:
        assert feats[col].map(type).eq(str).all()


def test_missing_raw_columns_handled():
    # пропущенные «сырые» колонки не должны ронять препроцессинг
    raw = _sample_raw(2).drop(columns=["dob", "merch_lat"])
    feats = run_preproc(raw)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert not feats[NUMERIC_FEATURES].isna().any().any()


def test_predict_scores_are_valid_probabilities():
    feats = run_preproc(_sample_raw(6))
    proba = predict_scores(feats)
    assert len(proba) == 6
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_predict_empty_input():
    feats = run_preproc(_sample_raw(1)).iloc[0:0]  # пустой фрейм с верными колонками
    proba = predict_scores(feats)
    assert len(proba) == 0


@pytest.mark.parametrize("score,expected", [(0.0, 0), (1.0, 1)])
def test_score_to_flag_bounds(score, expected):
    assert score_to_flag(score) == expected


def test_score_to_flag_uses_config_threshold():
    thr = float(load_feature_config()["threshold"])
    assert score_to_flag(thr + 1e-6) == 1
    assert score_to_flag(thr - 1e-6) == 0


def test_config_matches_preprocessing_contract():
    # защита от рассинхрона конфига модели и кода сервиса
    cfg = load_feature_config()
    assert cfg["feature_columns"] == FEATURE_COLUMNS
    assert cfg["categorical_features"] == CATEGORICAL_FEATURES
    assert cfg["raw_required_columns"] == RAW_REQUIRED_COLUMNS
