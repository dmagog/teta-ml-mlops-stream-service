"""Доп.сервис: консьюмер топика `scores` -> запись в витрину Postgres.

Отдельная consumer-группа. Запись идемпотентна (ON CONFLICT DO NOTHING),
подключения к Kafka и Postgres переживают недоступность на старте и в рантайме.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import psycopg2
from confluent_kafka import Consumer
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("db_writer")

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:29092")
SCORES_TOPIC = os.environ.get("SCORES_TOPIC", "scores")
CONSUMER_GROUP = os.environ.get("CONSUMER_GROUP", "db_writer")

PG_PARAMS = {
    "host": os.environ.get("POSTGRES_HOST", "postgres"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "dbname": os.environ.get("POSTGRES_DB", "fraud"),
    "user": os.environ.get("POSTGRES_USER", "fraud"),
    "password": os.environ.get("POSTGRES_PASSWORD", "fraud"),
}

INSERT_SQL = (
    "INSERT INTO transaction_scores "
    "(transaction_id, score, fraud_flag, us_state, merch, cat_id, amt, scored_at) VALUES %s "
    "ON CONFLICT (transaction_id) DO NOTHING"
)
BATCH_SIZE = 50


def connect_pg(retries: int = 30, delay: float = 3.0):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(**PG_PARAMS)
            conn.autocommit = True
            logger.info("Postgres подключён (%s:%s/%s).",
                        PG_PARAMS["host"], PG_PARAMS["port"], PG_PARAMS["dbname"])
            return conn
        except psycopg2.OperationalError as exc:
            last_err = exc
            logger.warning("Postgres недоступен %d/%d: %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError(f"Не удалось подключиться к Postgres: {last_err}")


def make_consumer(retries: int = 30, delay: float = 3.0) -> Consumer:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            consumer = Consumer({
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "group.id": CONSUMER_GROUP,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            })
            consumer.list_topics(timeout=5)
            consumer.subscribe([SCORES_TOPIC])
            logger.info("Kafka consumer готов (topic=%s, group=%s).", SCORES_TOPIC, CONSUMER_GROUP)
            return consumer
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Kafka недоступна %d/%d: %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError(f"Не удалось подключиться к Kafka: {last_err}")


def _to_row(r: dict) -> tuple:
    scored_at = r.get("scored_at")
    ts = datetime.fromisoformat(scored_at) if scored_at else datetime.now(timezone.utc)
    return (
        str(r.get("transaction_id", "")), float(r.get("score", 0.0)), int(r.get("fraud_flag", 0)),
        r.get("us_state"), r.get("merch"), r.get("cat_id"), float(r.get("amt") or 0.0), ts,
    )


def _flush(conn, rows: list[tuple]):
    """Записать батч; при обрыве соединения переподключиться и повторить один раз."""
    if not rows:
        return conn
    try:
        with conn.cursor() as cur:
            execute_values(cur, INSERT_SQL, rows)
        return conn
    except psycopg2.Error as exc:
        logger.warning("Сбой записи в Postgres, переподключаюсь: %s", exc)
        try:
            conn.close()
        except psycopg2.Error:
            pass
        conn = connect_pg()
        with conn.cursor() as cur:
            execute_values(cur, INSERT_SQL, rows)
        return conn


def main() -> None:
    logger.info("Запуск db_writer. topic=%s -> Postgres.transaction_scores", SCORES_TOPIC)
    conn = connect_pg()
    consumer = make_consumer()

    buffer: list[tuple] = []
    total = 0
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                if buffer:
                    conn = _flush(conn, buffer)
                    total += len(buffer)
                    buffer.clear()
                continue
            if msg.error():
                logger.error("Ошибка чтения: %s", msg.error())
                continue
            try:
                buffer.append(_to_row(json.loads(msg.value().decode("utf-8"))))
            except (ValueError, KeyError, TypeError) as exc:
                logger.error("Битая запись пропущена: %s", exc)
                continue

            if len(buffer) >= BATCH_SIZE:
                conn = _flush(conn, buffer)
                total += len(buffer)
                buffer.clear()
                if total % 500 == 0:
                    logger.info("Обработано из топика scores: %d", total)
    except KeyboardInterrupt:
        logger.info("Остановка по сигналу пользователя.")
    finally:
        conn = _flush(conn, buffer)
        consumer.close()
        conn.close()
        logger.info("db_writer остановлен. Всего обработано: %d", total + len(buffer))


if __name__ == "__main__":
    main()
