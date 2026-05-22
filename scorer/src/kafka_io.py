"""Этап пайплайна: ввод/вывод Kafka.

Создание продюсера/консьюмера обёрнуто в retry-цикл — depends_on/healthcheck
не гарантируют, что брокер уже принимает клиентов. Может запускаться как скрипт.
"""
from __future__ import annotations

import argparse
import logging
import time

from confluent_kafka import Consumer, Producer

from src.config import CONSUMER_GROUP, KAFKA_BOOTSTRAP, TRANSACTIONS_TOPIC

logger = logging.getLogger(__name__)


def make_producer(bootstrap: str = KAFKA_BOOTSTRAP, retries: int = 30, delay: float = 3.0) -> Producer:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            producer = Producer({"bootstrap.servers": bootstrap})
            producer.list_topics(timeout=5)
            logger.info("Kafka producer готов (bootstrap=%s).", bootstrap)
            return producer
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Kafka недоступна (producer) %d/%d: %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError(f"Не удалось подключиться к Kafka как producer: {last_err}")


def make_consumer(
    topic: str = TRANSACTIONS_TOPIC,
    group: str = CONSUMER_GROUP,
    bootstrap: str = KAFKA_BOOTSTRAP,
    retries: int = 30,
    delay: float = 3.0,
) -> Consumer:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            consumer = Consumer({
                "bootstrap.servers": bootstrap,
                "group.id": group,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            })
            consumer.list_topics(timeout=5)
            consumer.subscribe([topic])
            logger.info("Kafka consumer готов (topic=%s, group=%s).", topic, group)
            return consumer
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Kafka недоступна (consumer) %d/%d: %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError(f"Не удалось подключиться к Kafka как consumer: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test чтения топика Kafka.")
    parser.add_argument("--topic", default=TRANSACTIONS_TOPIC)
    parser.add_argument("--max", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    consumer = make_consumer(args.topic, group="smoke-test")
    seen = 0
    try:
        while seen < args.max:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Ошибка сообщения: %s", msg.error())
                continue
            print(msg.value().decode("utf-8"))
            seen += 1
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
