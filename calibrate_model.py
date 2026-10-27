"""Калибровка порога аномалий по benign-трафику.

Считает score (decision_function) на benign-строках переданных файлов,
выбирает порог по целевому FPR и сохраняет models/calibration.json.

Пример:
    python calibrate_model.py \
        --data Benign-Monday-no-metadata.parquet \
               Bruteforce-Tuesday-no-metadata.parquet \
               DoS-Wednesday-no-metadata.parquet \
        --target-fpr 0.02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.calibration import CALIBRATION_FILE
from modules.integrity import private_key_path, write_checksums
from modules.model_loader import (
    ModelLoadError,
    load_model_objects,
    load_normal_stats,
    resolve_model_dir,
)
from modules.pipeline import prepare_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Калибровка порога IsolationForest")
    parser.add_argument("--data", nargs="+", required=True, help="Parquet/CSV с колонкой Label")
    parser.add_argument("--model-dir", default=None, help="Каталог модели (по умолчанию models/)")
    parser.add_argument("--target-fpr", type=float, default=0.02, help="Целевая доля ложных срабатываний")
    parser.add_argument("--limit-per-file", type=int, default=120_000, help="Сколько benign-строк брать из файла")
    parser.add_argument("--output", default=None, help="Путь к calibration.json")
    return parser


def _read_benign(
    path: str, cols: list[str], limit: int, fill_values: pd.Series
) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path, columns=cols + ["Label"])
    else:
        df = pd.read_csv(path, usecols=cols + ["Label"])

    labels = df["Label"].astype(str).str.strip().str.lower()
    benign = df.loc[labels == "benign", cols]
    if benign.empty:
        raise ValueError(f"В {path} нет benign-строк.")
    if len(benign) > limit:
        benign = benign.sample(limit, random_state=42)
    # Та же обработка NaN/inf, что и при детекции: иначе порог считается
    # по распределению без «грязных» строк, а на них же потом применяется.
    X, _ = prepare_features(benign, cols, fill_values=fill_values)
    return X


def main() -> int:
    args = build_parser().parse_args()

    if not 0.0 < args.target_fpr < 1.0:
        print("Ошибка: --target-fpr должен быть в диапазоне (0, 1).")
        return 2

    try:
        model, scaler, selector, feature_cols = load_model_objects(args.model_dir)
        means, _ = load_normal_stats(feature_cols, model_dir=args.model_dir)
    except ModelLoadError as exc:
        print(f"Ошибка загрузки модели: {exc}")
        return 2

    all_scores = []
    per_file: dict[str, np.ndarray] = {}

    for path in args.data:
        if not os.path.isfile(path):
            print(f"Ошибка: файл не найден: {path}")
            return 2
        try:
            benign = _read_benign(path, feature_cols, args.limit_per_file, means)
        except Exception as exc:
            print(f"Ошибка чтения {path}: {exc}")
            return 2

        X_scaled = scaler.transform(benign)
        # В v2 селектора нет (см. model_loader.load_model_objects)
        X_filtered = selector.transform(X_scaled) if selector is not None else X_scaled
        scores = model.decision_function(X_filtered)
        per_file[os.path.basename(path)] = scores
        all_scores.append(scores)
        print(f"{os.path.basename(path)}: benign={len(benign):,}")

    merged = np.concatenate(all_scores)
    threshold = float(np.quantile(merged, args.target_fpr))
    achieved = {
        name: float((scores <= threshold).mean()) for name, scores in per_file.items()
    }

    print(f"\nВсего benign: {len(merged):,}")
    print(f"Порог (target FPR {args.target_fpr:.1%}): {threshold:.4f}")
    for name, fpr in achieved.items():
        print(f"  {name}: FPR {fpr:.2%}")

    model_dir = resolve_model_dir(args.model_dir)
    output = args.output or os.path.join(model_dir, CALIBRATION_FILE)
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    payload = {
        "threshold": threshold,
        "target_fpr": float(args.target_fpr),
        "achieved_fpr": achieved,
        "rows": int(len(merged)),
        "quantiles": [float(value) for value in np.quantile(merged, np.linspace(0, 1, 101))],
        "source": ", ".join(os.path.basename(path) for path in args.data),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\nСохранено: {os.path.abspath(output)}")

    # calibration.json входит в checksums.json: без пересчёта приложение
    # отклонит новый порог как подменённый.
    if os.path.abspath(output) == os.path.join(model_dir, CALIBRATION_FILE):
        if os.path.isfile(private_key_path()):
            write_checksums(model_dir)
            print("checksums.json пересчитан и подписан")
        else:
            print(
                "ВНИМАНИЕ: нет приватного ключа — пересчитайте и подпишите checksums: "
                "python make_checksums.py"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
