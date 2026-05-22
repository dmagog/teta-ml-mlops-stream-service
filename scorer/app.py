"""Ядро scorer-сервиса: потоковый скоринг транзакций из Kafka.

Объединяет три отдельных этапа: чтение/запись Kafka (src/kafka_io.py),
препроцессинг (src/preprocessing.py), инференс (src/scorer.py).
Транзакции скорятся микробатчами (по размеру батча или по таймауту-linger):
вход — топик `transactions`, выход — топик `scores` (3 обязательных поля
transaction_id/score/fraud_flag + аналитика для витрины и Grafana).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from src.config import SCORES_TOPIC, TRANSACTIONS_TOPIC, load_feature_config
from src.kafka_io import make_consumer, make_producer
from src.preprocessing import run_preproc
from src.scorer import predict_scores, score_to_flag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("scorer")

CONFIG = load_feature_config()
ID_COL = CONFIG["id_col"]
RAW_COLS = CONFIG["raw_required_columns"]
AMAP = CONFIG["analytics_columns"]

BATCH_SIZE = int(os.environ.get("SCORE_BATCH_SIZE", "200"))
FLUSH_INTERVAL = float(os.environ.get("SCORE_FLUSH_INTERVAL", "0.5"))


def _build_result(record: dict, score: float) -> dict:
    return {
        "transaction_id": str(record.get(ID_COL, "")),
        "score": round(float(score), 6),
        "fraud_flag": score_to_flag(float(score)),
        "us_state": record.get(AMAP["us_state"]),
        "merch": record.get(AMAP["merch"]),
        "cat_id": record.get(AMAP["cat_id"]),
        "amt": float(record.get(AMAP["amt"], 0.0) or 0.0),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def score_batch(records: list[dict]) -> list[dict]:
    raw = pd.DataFrame(records).reindex(columns=RAW_COLS)
    proba = predict_scores(run_preproc(raw, CONFIG))
    return [_build_result(rec, score) for rec, score in zip(records, proba)]


def score_records(records: list[dict]) -> list[dict]:
    """Батчевый скоринг с откатом на поштучный при битой записи."""
    if not records:
        return []
    try:
        return score_batch(records)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Сбой батч-скоринга (%s), откат на поштучный.", exc)
        results: list[dict] = []
        for rec in records:
            try:
                results.extend(score_batch([rec]))
            except Exception as inner:  # noqa: BLE001
                logger.error("Пропущена битая транзакция: %s", inner)
        return results


def _emit(producer, results: list[dict]) -> None:
    for res in results:
        producer.produce(SCORES_TOPIC, value=json.dumps(res).encode("utf-8"))
    producer.poll(0)


def main() -> None:
    logger.info("Запуск scorer. in=%s out=%s batch=%d", TRANSACTIONS_TOPIC, SCORES_TOPIC, BATCH_SIZE)
    consumer = make_consumer(TRANSACTIONS_TOPIC)
    producer = make_producer()

    batch: list[dict] = []
    processed = 0
    last_flush = time.monotonic()
    try:
        while True:
            msg = consumer.poll(0.2)
            if msg is not None:
                if msg.error():
                    logger.error("Ошибка чтения: %s", msg.error())
                else:
                    try:
                        batch.append(json.loads(msg.value().decode("utf-8")))
                    except (ValueError, AttributeError) as exc:
                        logger.error("Битое сообщение пропущено: %s", exc)

            now = time.monotonic()
            if len(batch) >= BATCH_SIZE or (batch and now - last_flush >= FLUSH_INTERVAL):
                _emit(producer, score_records(batch))
                processed += len(batch)
                batch.clear()
                last_flush = now
                if processed % 500 < BATCH_SIZE:
                    producer.flush(5)
                    logger.info("Обработано транзакций: %d", processed)
    except KeyboardInterrupt:
        logger.info("Остановка по сигналу пользователя.")
    finally:
        if batch:
            _emit(producer, score_records(batch))
            processed += len(batch)
        producer.flush(10)
        consumer.close()
        logger.info("Сервис остановлен. Всего обработано: %d", processed)


if __name__ == "__main__":
    main()
