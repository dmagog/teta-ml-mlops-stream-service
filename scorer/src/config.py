"""Пути, константы и настройки scorer-сервиса."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:29092")
TRANSACTIONS_TOPIC = os.environ.get("TRANSACTIONS_TOPIC", "transactions")
SCORES_TOPIC = os.environ.get("SCORES_TOPIC", "scores")
CONSUMER_GROUP = os.environ.get("CONSUMER_GROUP", "scorer")

THREAD_COUNT = int(os.environ.get("CATBOOST_THREAD_COUNT", "1"))


def load_feature_config(path: Path = FEATURE_CONFIG_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def model_path(config: dict | None = None) -> Path:
    config = config or load_feature_config()
    return MODELS_DIR / config["model_file"]
