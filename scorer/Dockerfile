FROM python:3.12-slim

WORKDIR /app

# Inference только на CPU: GPU отключён, число потоков CatBoost зафиксировано.
ENV CUDA_VISIBLE_DEVICES="" \
    CATBOOST_THREAD_COUNT=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY models/ ./models/

CMD ["python", "app.py"]
