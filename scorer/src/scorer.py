"""Этап пайплайна: инференс CatBoost (только CPU). Может запускаться как скрипт."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import THREAD_COUNT, load_feature_config, model_path

logger = logging.getLogger(__name__)


def load_model(config: dict | None = None) -> CatBoostClassifier:
    config = config or load_feature_config()
    model = CatBoostClassifier()
    model.load_model(str(model_path(config)))
    return model


_CONFIG = load_feature_config()
_MODEL = load_model(_CONFIG)
_THRESHOLD = float(_CONFIG["threshold"])


def predict_scores(features: pd.DataFrame, model: CatBoostClassifier | None = None) -> np.ndarray:
    """Вероятность фрода (класс 1) на CPU."""
    model = model or _MODEL
    if len(features) == 0:
        return np.empty(0, dtype=float)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def score_to_flag(score: float, threshold: float | None = None) -> int:
    return int(score >= (_THRESHOLD if threshold is None else threshold))


def main() -> None:
    parser = argparse.ArgumentParser(description="Скоринг признаков моделью.")
    parser.add_argument("features_path", type=Path, help="Parquet с признаками")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    proba = predict_scores(pd.read_parquet(args.features_path))
    flags = (proba >= _THRESHOLD).astype(int)
    print(f"threshold={_THRESHOLD:.3f} thread_count={THREAD_COUNT}")
    print(f"scores[:5]={np.round(proba[:5], 4)} fraud_flag sum={int(flags.sum())}/{len(flags)}")


if __name__ == "__main__":
    main()
