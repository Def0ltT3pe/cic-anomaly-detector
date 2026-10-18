"""Обучение supervised-детектора BruteForce.

IsolationForest не видит брутфорс (recall 0), потому что по per-flow
признакам он похож на обычные короткие сессии. Отдельный классификатор на
размеченных данных решает эту задачу.

Пример:
    python train_bruteforce.py \
        --data Bruteforce-Tuesday-no-metadata.parquet \
        --eval Benign-Monday-no-metadata.parquet DoS-Wednesday-no-metadata.parquet \
        --target-fpr 0.001
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.integrity import write_checksums
from modules.model_loader import load_model_objects, load_normal_stats, resolve_model_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Обучение BruteForce-классификатора")
    parser.add_argument("--data", nargs="+", required=True, help="Parquet/CSV с колонкой Label")
    parser.add_argument("--eval", nargs="*", default=[], help="Файлы для проверки FPR/recall")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def _read(path: str, cols: list[str], fill_values: pd.Series):
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path, columns=cols + ["Label"])
    else:
        df = pd.read_csv(path, usecols=cols + ["Label"])

    labels = df["Label"].astype(str).str.strip().str.lower()
    y = (labels != "benign").astype(int).to_numpy()
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    return X, y


def main() -> int:
    args = build_parser().parse_args()

    if not 0.0 < args.target_fpr < 1.0:
        print("Ошибка: --target-fpr должен быть в диапазоне (0, 1).")
        return 2

    try:
        _, _, _, feature_cols = load_model_objects()
        means, _ = load_normal_stats(feature_cols)
    except Exception as exc:
        print(f"Ошибка загрузки модели/статистик: {exc}")
        return 2

    frames, targets = [], []
    for path in args.data:
        if not os.path.isfile(path):
            print(f"Ошибка: файл не найден: {path}")
            return 2
        X, y = _read(path, feature_cols, means)
        frames.append(X)
        targets.append(y)
        print(f"{os.path.basename(path)}: строк {len(X):,}, атак {int(y.sum()):,}")

    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(targets)

    if len(np.unique(y)) < 2:
        print("Ошибка: в данных только один класс, классификатор не обучить.")
        return 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)
    print(f"Обучено: {len(X_train):,} строк, атак {int(y_train.sum()):,}")

    proba = model.predict_proba(X_test)[:, 1]
    benign_proba = proba[y_test == 0]
    threshold = float(np.quantile(benign_proba, 1.0 - args.target_fpr))
    predicted = proba >= threshold

    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "fpr": float((predicted[y_test == 0]).mean()),
        "attacks": int(y_test.sum()),
        "rows": int(len(y_test)),
    }
    print(
        "Holdout: AUC {auc:.4f}, recall {recall:.3f}, precision {precision:.3f}, FPR {fpr:.4%}".format(
            **metrics
        )
    )

    evaluation = {}
    for path in args.eval:
        if not os.path.isfile(path):
            print(f"ВНИМАНИЕ: файл для оценки не найден: {path}")
            continue
        X_eval, y_eval = _read(path, feature_cols, means)
        proba_eval = model.predict_proba(X_eval)[:, 1]
        predicted_eval = proba_eval >= threshold
        entry = {"rows": int(len(X_eval)), "attacks": int(y_eval.sum())}
        if (y_eval == 0).any():
            entry["fpr"] = float(predicted_eval[y_eval == 0].mean())
        if (y_eval == 1).any():
            entry["recall"] = float(predicted_eval[y_eval == 1].mean())
        evaluation[os.path.basename(path)] = entry
        print(f"  {os.path.basename(path)}: {entry}")

    output_dir = resolve_model_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "bruteforce.pkl")
    meta_path = os.path.join(output_dir, "bruteforce.json")
    joblib.dump(model, model_path)

    metadata = {
        "model": "RandomForestClassifier",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "target_fpr": float(args.target_fpr),
        "feature_cols": list(feature_cols),
        "training_files": [os.path.basename(path) for path in args.data],
        "rows": int(len(X)),
        "attacks": int(y.sum()),
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample",
            "seed": args.seed,
        },
        "metrics": metrics,
        "evaluation": evaluation,
        "note": (
            "Специализированный детектор BruteForce. Обучен на размеченных "
            "потоках CICIDS; для другой сети требуется переобучение."
        ),
    }
    with open(meta_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    write_checksums(output_dir)
    print(f"\nСохранено: {model_path}")
    print(f"Порог: {threshold:.4f}; checksums обновлены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
