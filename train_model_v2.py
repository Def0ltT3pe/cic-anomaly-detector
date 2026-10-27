"""Обучение экспериментальной модели v2 на benign-трафике нескольких дней.

Отличия от исходного dataset.py:
- benign берётся из нескольких дней (понедельник/вторник/среда), а не только
  из понедельника;
- нет no-op VarianceThreshold после StandardScaler (фильтр по дисперсии
  scale-dependent: на сырых данных он выкидывает редкие RST/ECE/FIN-флаги);
- порог калибруется по отложенному benign на целевой FPR.

Артефакты складываются в models_v2/ и по умолчанию не используются:
приложение читает models/, пока не задан MODEL_DIR=models_v2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.integrity import write_checksums
from modules.model_loader import load_model_objects
from modules.pipeline import prepare_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Обучение v2 на benign нескольких дней")
    parser.add_argument("--data", nargs="+", required=True, help="Parquet/CSV с колонкой Label")
    parser.add_argument("--eval", nargs="*", default=[], help="Файлы с атаками для оценки recall")
    parser.add_argument("--output-dir", default="models_v2")
    parser.add_argument("--limit-per-file", type=int, default=150_000)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument("--contamination", default="0.05")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def _read_columns(path: str, cols: list[str]) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=cols + ["Label"])
    return pd.read_csv(path, usecols=cols + ["Label"])


def _load_benign(path: str, cols: list[str], limit: int, seed: int) -> pd.DataFrame:
    """Сырые benign-строки: NaN/inf сохраняются и заполняются позже, как при детекции."""
    df = _read_columns(path, cols)
    labels = df["Label"].astype(str).str.strip().str.lower()
    benign = df.loc[labels == "benign", cols].replace([np.inf, -np.inf], np.nan)
    if benign.empty:
        raise ValueError(f"В {path} нет benign-строк.")
    if limit and len(benign) > limit:
        benign = benign.sample(limit, random_state=seed)
    return benign


def _load_attacks(path: str, cols: list[str], fill_values: pd.Series) -> pd.DataFrame:
    df = _read_columns(path, cols)
    labels = df["Label"].astype(str).str.strip().str.lower()
    attacks = df.loc[labels != "benign", cols]
    if attacks.empty:
        return attacks
    X, _ = prepare_features(attacks, cols, fill_values=fill_values)
    return X


def main() -> int:
    args = build_parser().parse_args()

    if not 0.0 < args.target_fpr < 1.0:
        print("Ошибка: --target-fpr должен быть в диапазоне (0, 1).")
        return 2

    try:
        contamination = float(args.contamination)
    except ValueError:
        contamination = args.contamination  # "auto"

    # Признаки берём из текущей модели, чтобы v2 была совместима по входу.
    _, _, _, cols = load_model_objects()
    print(f"Признаков: {len(cols)}")

    parts = []
    for path in args.data:
        if not os.path.isfile(path):
            print(f"Ошибка: файл не найден: {path}")
            return 2
        part = _load_benign(path, cols, args.limit_per_file, args.seed)
        parts.append(part)
        print(f"{os.path.basename(path)}: benign {len(part):,}")

    X = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    train_raw = X.iloc[order[:cut]]
    holdout_raw = X.iloc[order[cut:]]
    print(f"train={len(train_raw):,} holdout={len(holdout_raw):,}")

    # Статистики нормы — по конечным значениям train. Ими же заполняются
    # пропуски в train/holdout, ровно как pipeline.detect_anomalies делает
    # с входными данными (fill_values = normal_stats.json -> means).
    means = train_raw.mean()
    stds = train_raw.std()
    if not np.isfinite(means.to_numpy()).all():
        print("Ошибка: у части признаков в train нет ни одного конечного значения.")
        return 2
    if not np.isfinite(stds.to_numpy()).all() or (stds <= 0).any():
        print("Ошибка: в обучающей выборке есть нулевые/нефинитные std.")
        return 2

    train, train_info = prepare_features(train_raw, cols, fill_values=means)
    holdout, _ = prepare_features(holdout_raw, cols, fill_values=means)
    if train_info["nan_filled"]:
        print(f"Заполнено пропусков/inf средними нормы: {train_info['nan_filled']:,}")

    scaler = StandardScaler().fit(train)
    model = IsolationForest(
        contamination=contamination,
        random_state=args.seed,
        n_estimators=100,
        n_jobs=args.n_jobs,
    ).fit(scaler.transform(train))

    holdout_scores = model.decision_function(scaler.transform(holdout))
    threshold = float(np.quantile(holdout_scores, args.target_fpr))
    holdout_fpr = float((holdout_scores <= threshold).mean())
    print(f"Порог {threshold:.4f}; FPR на holdout {holdout_fpr:.2%}")

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(BASE_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(list(cols), os.path.join(output_dir, "feature_cols_filtered.pkl"))

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats_payload = {
        "source": ", ".join(os.path.basename(path) for path in args.data),
        "rows": int(len(train)),
        "created_at": created_at,
        "feature_cols": list(cols),
        "means": {col: float(value) for col, value in means.items()},
        "stds": {col: float(value) for col, value in stds.items()},
    }
    with open(
        os.path.join(output_dir, "normal_stats.json"), "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(stats_payload, handle, ensure_ascii=False, indent=2)

    calibration_payload = {
        "threshold": threshold,
        "target_fpr": float(args.target_fpr),
        "achieved_fpr": {"holdout": holdout_fpr},
        "rows": int(len(holdout)),
        "quantiles": [float(value) for value in np.quantile(holdout_scores, np.linspace(0, 1, 101))],
        "source": "v2 holdout",
        "created_at": created_at,
    }
    with open(
        os.path.join(output_dir, "calibration.json"), "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(calibration_payload, handle, ensure_ascii=False, indent=2)

    evaluation = {}
    for path in args.eval:
        if not os.path.isfile(path):
            print(f"ВНИМАНИЕ: файл для оценки не найден: {path}")
            continue
        attacks = _load_attacks(path, cols, means)
        if attacks.empty:
            continue
        scores = model.decision_function(scaler.transform(attacks))
        recall = float((scores <= threshold).mean())
        evaluation[os.path.basename(path)] = {
            "attacks": int(len(attacks)),
            "recall": recall,
            "flagged": int((scores <= threshold).sum()),
        }
        print(f"{os.path.basename(path)}: атак {len(attacks):,}, recall {recall:.3f}")

    metadata = {
        "version": "v2-experimental",
        "created_at": created_at,
        "note": (
            "Экспериментальная модель: обучена на benign нескольких дней, без селектора. "
            "По AUC не превосходит models/; по умолчанию не используется."
        ),
        "features": len(cols),
        "training_files": [os.path.basename(path) for path in args.data],
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "contamination": contamination,
        "n_estimators": 100,
        "target_fpr": float(args.target_fpr),
        "threshold": threshold,
        "holdout_fpr": holdout_fpr,
        "evaluation": evaluation,
    }
    with open(
        os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    write_checksums(output_dir)

    print(f"\nСохранено в {output_dir}")
    print("Включить: MODEL_DIR=models_v2 (или --model-dir models_v2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
