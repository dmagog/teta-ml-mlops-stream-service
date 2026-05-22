"""Streamlit UI: имитация потока транзакций и просмотр результатов скоринга."""
from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from confluent_kafka import Producer
from sqlalchemy import create_engine, text

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:29092")
TRANSACTIONS_TOPIC = os.environ.get("TRANSACTIONS_TOPIC", "transactions")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DEFAULT_CSV = os.path.join(DATA_DIR, "test.csv")

PG_PARAMS = {
    "host": os.environ.get("POSTGRES_HOST", "postgres"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "dbname": os.environ.get("POSTGRES_DB", "fraud"),
    "user": os.environ.get("POSTGRES_USER", "fraud"),
    "password": os.environ.get("POSTGRES_PASSWORD", "fraud"),
}


@st.cache_resource
def get_producer() -> Producer:
    return Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})


@st.cache_resource
def get_engine():
    url = (
        f"postgresql+psycopg2://{PG_PARAMS['user']}:{PG_PARAMS['password']}"
        f"@{PG_PARAMS['host']}:{PG_PARAMS['port']}/{PG_PARAMS['dbname']}"
    )
    return create_engine(url, pool_pre_ping=True)


def pg_query(sql: str) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), get_engine())


def page_feed() -> None:
    st.header("Подача данных в поток")
    st.caption(f"Транзакции отправляются в топик Kafka **{TRANSACTIONS_TOPIC}**.")

    uploaded = st.file_uploader("CSV формата test.csv (необязательно)", type=["csv"])
    if uploaded is not None:
        df, source = pd.read_csv(uploaded), uploaded.name
    elif os.path.exists(DEFAULT_CSV):
        df, source = pd.read_csv(DEFAULT_CSV), DEFAULT_CSV
    else:
        st.warning("Нет данных: загрузите CSV или положите test.csv в смонтированную папку.")
        return

    st.write(f"Источник: `{source}` — строк: **{len(df)}**")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    rate = col1.slider("Скорость, транзакций/сек", 1, 500, 50)
    n_rows = col2.number_input("Сколько отправить", 1, len(df), min(len(df), 1000))

    if st.button("Запустить поток", type="primary"):
        producer = get_producer()
        progress = st.progress(0.0)
        status = st.empty()
        subset = df.head(int(n_rows))
        delay = 1.0 / rate
        for sent, (_, row) in enumerate(subset.iterrows(), start=1):
            producer.produce(TRANSACTIONS_TOPIC, value=json.dumps(row.to_dict(), default=str).encode())
            producer.poll(0)
            if sent % 25 == 0 or sent == len(subset):
                progress.progress(sent / len(subset))
                status.write(f"Отправлено: {sent}/{len(subset)}")
            time.sleep(delay)
        producer.flush(10)
        st.success(f"Готово. Отправлено {len(subset)} транзакций в топик «{TRANSACTIONS_TOPIC}».")


def page_results() -> None:
    st.header("Результаты скоринга")
    st.caption("Данные из витрины Postgres `transaction_scores`.")

    if not st.button("Посмотреть результаты", type="primary"):
        st.info("Нажмите кнопку, чтобы загрузить последние результаты из базы.")
        return

    try:
        total = int(pg_query("SELECT count(*) AS n FROM transaction_scores")["n"].iloc[0])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось обратиться к Postgres: {exc}")
        return

    st.metric("Всего транзакций в витрине", total)
    if total == 0:
        st.warning("В витрине пока нет данных — запустите поток на странице «Подача данных».")
        return

    st.subheader("Последние 10 фрод-транзакций (fraud_flag = 1)")
    fraud = pg_query(
        "SELECT transaction_id AS \"ID транзакции\", round(score::numeric, 4) AS score, "
        "fraud_flag, us_state AS \"штат\", merch AS \"мерчант\", cat_id AS \"категория\", "
        "round(amt::numeric, 2) AS \"сумма\", "
        "to_char(scored_at, 'YYYY-MM-DD HH24:MI:SS') AS \"время (UTC)\" "
        "FROM transaction_scores WHERE fraud_flag = 1 ORDER BY scored_at DESC LIMIT 10"
    )
    if fraud.empty:
        st.write("Фрод-транзакций пока не обнаружено.")
    else:
        st.dataframe(fraud, use_container_width=True, hide_index=True)

    st.subheader("Распределение скоров последних 100 транзакций")
    scores = pg_query(
        "SELECT score FROM (SELECT score, scored_at FROM transaction_scores "
        "ORDER BY scored_at DESC LIMIT 100) t"
    )["score"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    weights = np.ones(len(scores)) / max(len(scores), 1)
    ax.hist(scores, bins=20, range=(0, 1), weights=weights, color="#d2451e", edgecolor="white")
    ax.set_xlabel("score (вероятность фрода)")
    ax.set_ylabel("доля транзакций")
    ax.set_title(f"Нормированная плотность скоров (n={len(scores)})")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Fraud Stream — TETA MLOps", layout="wide")
    st.title("Fraud Scoring Stream")
    page = st.sidebar.radio("Раздел", ["Подача данных", "Посмотреть результаты"])
    st.sidebar.caption(f"Kafka: {KAFKA_BOOTSTRAP}\n\nPostgres: {PG_PARAMS['host']}:{PG_PARAMS['port']}")
    page_feed() if page == "Подача данных" else page_results()


if __name__ == "__main__":
    main()
