"""Headless-продюсер транзакций в Kafka (для smoke-теста и воспроизводимого демо).

С хоста (Kafka проброшена на localhost:9092):
    pip install confluent-kafka pandas
    python scripts/produce.py --limit 1000 --rate 100
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer


def main() -> None:
    parser = argparse.ArgumentParser(description="Подача транзакций из CSV в топик Kafka.")
    parser.add_argument("--csv", default="examples/test.csv")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--rate", type=float, default=100.0, help="транзакций/сек")
    parser.add_argument("--limit", type=int, default=0, help="0 = весь файл")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    producer = Producer({"bootstrap.servers": args.bootstrap})
    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    for sent, (_, row) in enumerate(df.iterrows(), start=1):
        producer.produce(args.topic, value=json.dumps(row.to_dict(), default=str).encode())
        producer.poll(0)
        if delay:
            time.sleep(delay)
    producer.flush(15)
    print(f"Отправлено {len(df)} транзакций в топик '{args.topic}' ({args.bootstrap}).")


if __name__ == "__main__":
    main()
