-- Витрина результатов скоринга. Создаётся автоматически при первом старте
-- контейнера postgres (файл смонтирован в /docker-entrypoint-initdb.d/).
CREATE TABLE IF NOT EXISTS transaction_scores (
    id            BIGSERIAL PRIMARY KEY,
    transaction_id TEXT             NOT NULL,
    score         DOUBLE PRECISION  NOT NULL,
    fraud_flag    SMALLINT          NOT NULL,
    us_state      TEXT,
    merch         TEXT,
    cat_id        TEXT,
    amt           DOUBLE PRECISION,
    scored_at     TIMESTAMPTZ       NOT NULL DEFAULT now(),
    -- Идемпотентность: повторная подача той же транзакции (или реплей из Kafka)
    -- не создаёт дубликат (db_writer пишет с ON CONFLICT DO NOTHING).
    CONSTRAINT uq_transaction_id UNIQUE (transaction_id)
);

-- Индексы под запросы UI (последние fraud-записи) и Grafana (фильтры, TPS, barplot).
CREATE INDEX IF NOT EXISTS idx_scores_scored_at  ON transaction_scores (scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_scores_fraud_flag ON transaction_scores (fraud_flag);
CREATE INDEX IF NOT EXISTS idx_scores_us_state   ON transaction_scores (us_state);
CREATE INDEX IF NOT EXISTS idx_scores_merch      ON transaction_scores (merch);
CREATE INDEX IF NOT EXISTS idx_scores_cat_id     ON transaction_scores (cat_id);
