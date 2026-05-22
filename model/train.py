"""Обучение лёгкого CatBoostClassifier для фрод-скоринга (CPU).

Использует тот же препроцессинг, что и сервис, и сохраняет в scorer/models/
модель (fraud_catboost.cbm) и feature_config.json (фичи, порог, маппинг колонок).
Запуск: python model/train.py  (после python model/generate_data.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scorer.src.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    RAW_REQUIRED_COLUMNS,
    run_preproc,
)

TARGET = "is_fraud"
ID_COL = "trans_num"


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Порог, максимизирующий F1 на валидации."""
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение CatBoost фрод-классификатора.")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "model" / "data" / "train.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "scorer" / "models")
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    X = run_preproc(df)
    y = df[TARGET].astype(int).values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_pool = Pool(X_tr, y_tr, cat_features=CATEGORICAL_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_FEATURES)

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5,
        random_seed=42,
        thread_count=-1,
        task_type="CPU",
        verbose=100,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    threshold = _best_threshold(y_val, proba)
    pred = (proba >= threshold).astype(int)
    print(
        f"\nVAL  AUC={auc:.4f}  threshold={threshold:.3f}  "
        f"precision={precision_score(y_val, pred, zero_division=0):.3f}  "
        f"recall={recall_score(y_val, pred, zero_division=0):.3f}  "
        f"F1={f1_score(y_val, pred, zero_division=0):.3f}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "fraud_catboost.cbm"
    model.save_model(str(model_path))

    config = {
        "model_file": "fraud_catboost.cbm",
        "id_col": ID_COL,
        "threshold": threshold,
        "val_auc": round(float(auc), 4),
        "raw_required_columns": RAW_REQUIRED_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "analytics_columns": {
            "us_state": "state",
            "merch": "merchant",
            "cat_id": "category",
            "amt": "amt",
        },
    }
    config_path = args.out_dir / "feature_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\nМодель  -> {model_path}")
    print(f"Конфиг  -> {config_path}")


if __name__ == "__main__":
    main()
