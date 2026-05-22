"""Генератор синтетического датасета транзакций в схеме sparkov.

Данные синтетические, но с заложенным сигналом фрода (растёт с суммой, для
онлайн-категорий, ночью и при большой дистанции карта-мерчант), поэтому модель
реально обучается. Создаёт train.csv (с is_fraud) и test.csv (без метки, для UI).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATES = [
    "NY", "CA", "TX", "FL", "IL", "PA", "OH", "GA",
    "NC", "MI", "NJ", "VA", "WA", "AZ", "MA",
]
STATE_GEO = {  # центры штатов (lat, lon)
    "NY": (42.9, -75.5), "CA": (36.7, -119.4), "TX": (31.0, -99.0),
    "FL": (27.8, -81.7), "IL": (40.0, -89.0), "PA": (41.2, -77.2),
    "OH": (40.4, -82.8), "GA": (32.7, -83.2), "NC": (35.6, -79.0),
    "MI": (44.3, -85.6), "NJ": (40.1, -74.5), "VA": (37.5, -78.7),
    "WA": (47.4, -120.5), "AZ": (34.2, -111.7), "MA": (42.3, -71.8),
}
CATEGORIES = [
    "grocery_pos", "gas_transport", "misc_net", "shopping_net", "shopping_pos",
    "entertainment", "food_dining", "health_fitness", "home", "kids_pets",
    "misc_pos", "personal_care", "travel", "grocery_net",
]
ONLINE_CATEGORIES = {"misc_net", "shopping_net", "grocery_net"}
MERCHANTS = [
    "Rippin-Kub", "Heller-Langosh", "Schumm PLC", "Kuhn LLC", "Predovic Inc",
    "Kihn-Gusikowski", "Goyette Inc", "Lind-Buckridge", "Stokes LLC", "Cormier LLC",
    "Padberg-Welch", "Hahn-Douglas", "Romaguera Ltd", "Stracke-Lemke", "Kunze Inc",
    "Bauch-Raynor", "Doyle Ltd", "Sporer-Keebler", "Konopelski Group", "Beier LLC",
]
GENDERS = ["M", "F"]
JOBS = [
    "Engineer", "Teacher", "Nurse", "Designer", "Accountant", "Driver",
    "Manager", "Analyst", "Chef", "Electrician", "Pharmacist", "Lawyer",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    state = rng.choice(STATES, size=n)
    category = rng.choice(CATEGORIES, size=n)
    merchant = rng.choice(MERCHANTS, size=n)
    gender = rng.choice(GENDERS, size=n)
    job = rng.choice(JOBS, size=n)

    amt = np.round(rng.lognormal(mean=3.2, sigma=1.1, size=n), 2)

    base = pd.Timestamp("2025-05-01")
    offsets = rng.integers(0, 30 * 24 * 3600, size=n)
    ts = base + pd.to_timedelta(offsets, unit="s")
    hour = ts.hour

    lat0 = np.array([STATE_GEO[s][0] for s in state])
    lon0 = np.array([STATE_GEO[s][1] for s in state])
    lat = np.round(lat0 + rng.normal(0, 0.5, n), 4)
    lon = np.round(lon0 + rng.normal(0, 0.5, n), 4)
    merch_lat = np.round(lat + rng.normal(0, 0.3, n), 4)
    merch_long = np.round(lon + rng.normal(0, 0.3, n), 4)

    city_pop = rng.integers(1_000, 3_000_000, size=n)
    dob = pd.Timestamp("1990-01-01") - pd.to_timedelta(rng.integers(0, 60 * 365, size=n), unit="D")

    # Сигнал фрода: онлайн-категории, ночь, аномально далёкий мерчант, крупная сумма.
    is_online = np.isin(category, list(ONLINE_CATEGORIES)).astype(float)
    is_night = ((hour <= 4) | (hour >= 22)).astype(float)
    far_jump = rng.random(n) < 0.12
    merch_lat = np.where(far_jump, np.round(lat + rng.normal(0, 4.0, n), 4), merch_lat)
    merch_long = np.where(far_jump, np.round(lon + rng.normal(0, 4.0, n), 4), merch_long)
    dist = np.sqrt((merch_lat - lat) ** 2 + (merch_long - lon) ** 2)
    big_dist = (dist > 1.5).astype(float)

    logit = (
        -4.0
        + 2.0 * (amt > 150).astype(float)
        + 1.8 * is_online
        + 1.6 * is_night
        + 2.4 * big_dist
    )
    p_fraud = _sigmoid(logit)
    is_fraud = (rng.random(n) < p_fraud).astype(int)

    df = pd.DataFrame(
        {
            "trans_num": [f"t{seed}_{i:08d}" for i in range(n)],
            "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": rng.integers(10**15, 10**16, size=n),
            "merchant": merchant,
            "category": category,
            "amt": amt,
            "gender": gender,
            "city": ["City_" + s for s in state],
            "state": state,
            "zip": rng.integers(10000, 99999, size=n),
            "lat": lat,
            "long": lon,
            "city_pop": city_pop,
            "job": job,
            "dob": dob.strftime("%Y-%m-%d"),
            "unix_time": (ts.astype("int64") // 10**9),
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "is_fraud": is_fraud,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация синтетического sparkov-датасета.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--examples-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "examples")
    parser.add_argument("--n-train", type=int, default=60_000)
    parser.add_argument("--n-test", type=int, default=2_000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.examples_dir.mkdir(parents=True, exist_ok=True)

    train = generate(args.n_train, seed=42)
    test = generate(args.n_test, seed=7).drop(columns=["is_fraud"])

    train_path = args.out_dir / "train.csv"
    test_path = args.examples_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"train -> {train_path}  ({len(train)} строк, фрод={train['is_fraud'].mean():.3%})")
    print(f"test  -> {test_path}  ({len(test)} строк, без метки)")


if __name__ == "__main__":
    main()
