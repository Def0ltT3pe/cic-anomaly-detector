"""Дообучение и калибровка модели под трафик целевой сети.

Модели, обученные на CIC-IDS2017, плохо переносятся на другие сети (см.
evaluate.py): на CSE-CIC-IDS2018 AUC IsolationForest 0.44, а FPR уходит от
целевых 2% в 0.2-10%. Внутри одной сети те же алгоритмы работают хорошо,
поэтому при развёртывании модель нужно подстроить под сеть:

1. IsolationForest переобучается на benign-трафике целевой сети, порог и
   normal_stats.json считаются по отложенной части того же трафика;
2. BruteForce-классификатор:
   - есть размеченные брутфорс-атаки (колонка Label) — переобучается;
   - нет — у базовой модели пересчитывается порог под FPR этой сети;
3. результат — каталог модели в формате models/ (подписанный checksums.json,
   если есть ключ), включается через MODEL_DIR=<каталог>.

Отчёт сравнивает базовую и адаптированную модели на одном и том же
отложенном трафике целевой сети.

Входные данные — CSV/Parquet от CICFlowMeter с теми же признаками, что у
базовой модели. Без колонки Label весь трафик считается эталоном нормы:
записывайте его в спокойный период; --trim отбросит самые аномальные
строки, если в запись могли попасть атаки или сканирования.

Пример:
    python adapt_model.py --data site/*.parquet --output-dir models_site
    MODEL_DIR=models_site streamlit run App.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluate import Dataset, attack_family, evaluate_scores
from modules.bruteforce import BruteForceError, load_bruteforce
from modules.calibration import CALIBRATION_FILE, CalibrationError, load_calibration
from modules.integrity import CHECKSUMS_FILE, public_key_path, write_checksums
from modules.model_loader import (
    STATS_FILE,
    ModelLoadError,
    load_model_objects,
    load_normal_stats,
    resolve_model_dir,
)
from modules.pipeline import FeatureValidationError, prepare_features, signed_log1p

MIN_BENIGN_TRAIN = 1_000
MIN_BENIGN_HOLDOUT = 200
MIN_ATTACK_TRAIN = 50
BRUTE_FAMILY = "BruteForce"
CLASSIFIER_NAMES = {"all": "Атаки", "bruteforce": "BruteForce"}


class AdaptationError(RuntimeError):
    """Данных недостаточно или они не подходят для адаптации."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Дообучение и калибровка под целевую сеть")
    parser.add_argument("--data", nargs="+", required=True, help="CSV/Parquet целевой сети")
    parser.add_argument("--output-dir", required=True, help="Каталог новой модели")
    parser.add_argument("--base-dir", default=None, help="Базовая модель (по умолчанию models/)")
    parser.add_argument("--force", action="store_true", help="Перезаписать --output-dir")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument(
        "--classifier",
        choices=("all", "bruteforce"),
        default="all",
        help="Чему учить supervised-модель на размеченных данных: all — отличать "
        "любые атаки от нормы (точнее всего внутри сети), bruteforce — только "
        "брутфорс, как базовая модель. Без меток базовая модель только перекалибруется.",
    )
    parser.add_argument("--target-fpr", type=float, default=0.02, help="FPR для IsolationForest")
    parser.add_argument(
        "--brute-target-fpr", type=float, default=0.001, help="FPR для supervised-модели"
    )
    parser.add_argument("--holdout", type=float, default=0.2, help="Доля для калибровки и отчёта")
    parser.add_argument(
        "--split",
        choices=("time", "random"),
        default="time",
        help="time — holdout из конца каждого файла (честнее: соседние потоки "
        "одной сессии не попадают в обе части); random — случайный",
    )
    parser.add_argument("--limit-per-file", type=int, default=300_000)
    parser.add_argument(
        "--trim",
        type=float,
        default=0.0,
        help="Доля самых аномальных benign-строк, которые выбрасываются перед "
        "финальным обучением IF (если эталон может быть загрязнён)",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        help="sign·log1p перед StandardScaler: выше AUC, но ниже recall при малом FPR",
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="Деревьев в IsolationForest")
    # Подобрано на CSE-CIC-IDS2018 (1.2 млн потоков): по сравнению с прежними
    # 60 деревьями глубины 12 пропусков атак (без Infiltration) при FPR 0.1%
    # меньше в ~2.5 раза (1.7% -> 0.7%), модель ~36 МБ вместо ~4.5 МБ.
    # Без ограничения глубины точнее ещё на 0.3 п.п., но весит ~213 МБ.
    parser.add_argument("--rf-trees", type=int, default=100, help="Деревьев в supervised RF")
    parser.add_argument(
        "--rf-depth", type=int, default=20, help="Макс. глубина RF (0 — без ограничения)"
    )
    parser.add_argument("--rf-leaf", type=int, default=5, help="min_samples_leaf у RF")
    parser.add_argument(
        "--rf-max-samples",
        type=float,
        default=0.5,
        help="Доля строк в бутстрепе каждого дерева (меньше — быстрее и легче модель)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def _expand(patterns: list[str]) -> list[str]:
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        files.extend(matches)
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise AdaptationError("Файлы не найдены: " + ", ".join(missing))
    return files


def _read_file(path: str, cols: list[str], label_column: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif lower.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        raise AdaptationError(f"{path}: поддерживаются только .csv и .parquet")
    # В исходных CSV CICFlowMeter у части колонок ведущие пробелы
    df.columns = [str(col).strip() for col in df.columns]
    keep = [col for col in cols if col in df.columns]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise AdaptationError(
            f"{os.path.basename(path)}: нет {len(missing)} признаков базовой модели "
            f"({', '.join(missing[:5])}…)"
        )
    if label_column in df.columns:
        keep.append(label_column)
    return df[keep]


def load_target(
    files: list[str],
    cols: list[str],
    label_column: str,
    holdout: float,
    split: str,
    limit: int,
    seed: int,
):
    """Читает файлы и делит каждый на train/holdout.

    Возвращает (frame, labels, is_holdout, labeled). Без колонки Label все
    строки файла считаются benign.
    """
    rng = np.random.default_rng(seed)
    frames, labels, holdout_flags = [], [], []
    labeled = False
    for path in files:
        df = _read_file(path, cols, label_column)
        if limit and len(df) > limit:
            # Выборка с сохранением порядка, чтобы time-split оставался временным
            keep = np.sort(rng.choice(len(df), limit, replace=False))
            df = df.iloc[keep]
        df = df.reset_index(drop=True)

        if label_column in df.columns:
            labeled = True
            file_labels = df.pop(label_column).astype(str).str.strip()
        else:
            file_labels = pd.Series(["Benign"] * len(df))

        n_holdout = int(round(len(df) * holdout))
        flags = np.zeros(len(df), dtype=bool)
        if split == "time":
            flags[len(df) - n_holdout:] = True
        else:
            flags[rng.choice(len(df), n_holdout, replace=False)] = True

        frames.append(df)
        labels.append(file_labels)
        holdout_flags.append(flags)
        print(f"{os.path.basename(path)}: {len(df):,} строк")

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return (
        frame,
        pd.concat(labels, ignore_index=True),
        np.concatenate(holdout_flags),
        labeled,
    )


def _if_scores(model, scaler, selector, X: pd.DataFrame) -> np.ndarray:
    """decision_function: меньше — аномальнее (как в pipeline.detect_anomalies)."""
    scaled = scaler.transform(X)
    if selector is not None:
        scaled = selector.transform(scaled)
    return np.asarray(model.decision_function(scaled), dtype=float)


def _strict_threshold(quantile: float) -> float:
    """BruteForceDetector помечает proba >= threshold. У RF много одинаковых
    вероятностей (часто 0), поэтому порог чуть выше квантиля: иначе квантиль 0
    пометил бы весь трафик."""
    return float(np.nextafter(quantile, np.inf))


def _summary(metrics: dict) -> dict:
    keys = ("rows", "attacks", "auc", "fpr", "recall", "families")
    return {key: metrics.get(key) for key in keys}


def adapt(args) -> dict:
    base_dir = resolve_model_dir(args.base_dir)
    output_dir = resolve_model_dir(args.output_dir)
    if output_dir == base_dir:
        raise AdaptationError("--output-dir совпадает с базовой моделью: укажите новый каталог.")
    if os.path.exists(output_dir) and os.listdir(output_dir) and not args.force:
        raise AdaptationError(f"{output_dir} уже существует. Добавьте --force, чтобы перезаписать.")

    base_model, base_scaler, base_selector, cols = load_model_objects(base_dir)
    base_means, base_stds = load_normal_stats(cols, model_dir=base_dir)
    try:
        base_calibration = load_calibration(base_dir)
    except CalibrationError as exc:
        print(f"ВНИМАНИЕ: калибровка базовой модели не загружена ({exc})")
        base_calibration = None
    try:
        base_brute = load_bruteforce(base_dir)
    except BruteForceError as exc:
        print(f"ВНИМАНИЕ: BruteForce базовой модели не загружен ({exc})")
        base_brute = None

    files = _expand(args.data)
    frame, labels, is_holdout, labeled = load_target(
        files, cols, args.label_column, args.holdout, args.split, args.limit_per_file, args.seed
    )
    is_benign = (labels.str.lower() == "benign").to_numpy()
    benign_train = np.flatnonzero(is_benign & ~is_holdout)
    benign_holdout = np.flatnonzero(is_benign & is_holdout)
    holdout_idx = np.flatnonzero(is_holdout)

    if len(benign_train) < MIN_BENIGN_TRAIN:
        raise AdaptationError(
            f"Мало benign-строк для обучения: {len(benign_train):,} (нужно ≥ {MIN_BENIGN_TRAIN:,})."
        )
    if len(benign_holdout) < MIN_BENIGN_HOLDOUT:
        raise AdaptationError(
            f"Мало benign-строк для калибровки: {len(benign_holdout):,} "
            f"(нужно ≥ {MIN_BENIGN_HOLDOUT:,}). Увеличьте данные или --holdout."
        )
    if not labeled:
        print("Колонки Label нет: весь трафик считается эталоном нормы.")

    # --- Статистики нормы целевой сети (заполнение пропусков и объяснения)
    raw_benign = frame.iloc[benign_train]
    means = raw_benign.mean()
    stds = raw_benign.std()
    means = means.where(np.isfinite(means), base_means)
    # Константный в этой сети признак: std=0 сломает объяснения — берём базовый
    replaced_std = [col for col in cols if not (np.isfinite(stds[col]) and stds[col] > 0)]
    stds = stds.where(np.isfinite(stds) & (stds > 0), base_stds)

    X, info = prepare_features(frame, cols, fill_values=means)
    if info["nan_filled"]:
        print(f"Заполнено пропусков/inf средними нормы сети: {info['nan_filled']:,}")

    # --- IsolationForest на benign целевой сети
    def make_scaler():
        if args.log1p:
            return make_pipeline(
                FunctionTransformer(signed_log1p, feature_names_out="one-to-one"),
                StandardScaler(),
            )
        return StandardScaler()

    def fit_if(rows: np.ndarray):
        scaler = make_scaler().fit(X.iloc[rows])
        model = IsolationForest(
            n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs
        ).fit(scaler.transform(X.iloc[rows]))
        return scaler, model

    train_rows = benign_train
    scaler, model = fit_if(train_rows)
    trimmed = 0
    if args.trim > 0:
        train_scores = _if_scores(model, scaler, None, X.iloc[train_rows])
        cut = np.quantile(train_scores, args.trim)
        keep = train_rows[train_scores > cut]
        trimmed = len(train_rows) - len(keep)
        train_rows = keep
        scaler, model = fit_if(train_rows)
        print(f"Отброшено самых аномальных benign-строк: {trimmed:,}")

    holdout_benign_scores = _if_scores(model, scaler, None, X.iloc[benign_holdout])
    threshold = float(np.quantile(holdout_benign_scores, args.target_fpr))
    holdout_fpr = float((holdout_benign_scores <= threshold).mean())

    # --- Сравнение с базовой моделью на одном и том же holdout цели
    holdout_data = Dataset(
        name="target", X=X, labels=labels, files=pd.Series(["target"] * len(X))
    )
    X_base, _ = prepare_features(frame.iloc[holdout_idx], cols, fill_values=base_means)
    base_scores = _if_scores(base_model, base_scaler, base_selector, X_base)
    # Без калибровки базовая модель работает через predict(): аномалия при score < 0
    base_threshold = base_calibration.threshold if base_calibration is not None else 0.0
    adapted_scores = _if_scores(model, scaler, None, X.iloc[holdout_idx])
    # evaluate_scores считает «больше — аномальнее» и flagged = score > threshold
    evaluation = {
        "isolation_forest": {
            "base": _summary(
                evaluate_scores(holdout_data, -base_scores, -base_threshold, args.target_fpr, holdout_idx)
            ),
            "adapted": _summary(
                evaluate_scores(holdout_data, -adapted_scores, -threshold, args.target_fpr, holdout_idx)
            ),
        }
    }

    # --- Покрытие: какие семейства атак есть в train и holdout
    families = labels.map(attack_family).to_numpy()
    coverage = {}
    for family in sorted(set(families) - {"Benign"}):
        mask = families == family
        coverage[family] = {
            "train": int((mask & ~is_holdout).sum()),
            "holdout": int((mask & is_holdout).sum()),
        }
    unchecked = [f for f, c in coverage.items() if c["train"] and not c["holdout"]]

    # --- Supervised-модель (слот bruteforce.pkl/json)
    brute_payload = None
    brute_model = None
    if base_brute is not None:
        feature_cols = list(base_brute.feature_cols)
        if args.classifier == "all":
            positive = ~is_benign
        else:
            positive = families == BRUTE_FAMILY
        positive_train = int((positive & ~is_holdout).sum())
        base_proba = base_brute.predict_proba(X_base)

        if positive_train >= MIN_ATTACK_TRAIN:
            mode = "retrained"
            display_name = CLASSIFIER_NAMES[args.classifier]
            train_rows_rf = np.flatnonzero(~is_holdout)
            rf_params = {
                "n_estimators": args.rf_trees,
                "max_depth": args.rf_depth or None,
                "min_samples_leaf": args.rf_leaf,
                "max_samples": args.rf_max_samples,
                "class_weight": "balanced_subsample",
                "seed": args.seed,
            }
            brute_model = RandomForestClassifier(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                min_samples_leaf=rf_params["min_samples_leaf"],
                max_samples=rf_params["max_samples"],
                class_weight=rf_params["class_weight"],
                n_jobs=args.n_jobs,
                random_state=args.seed,
            ).fit(X.iloc[train_rows_rf][feature_cols], positive[train_rows_rf].astype(int))
            # Train RF уже видел — порог по benign holdout
            calib_rows = benign_holdout
        else:
            mode = "recalibrated"
            display_name = base_brute.display_name
            brute_model = base_brute.model
            rf_params = base_brute.metadata.get("params")
            # Модель эту сеть не видела — порог по benign train, holdout честный
            calib_rows = benign_train
            if labeled and args.classifier == "all":
                print(
                    f"Размеченных атак в train {positive_train} < {MIN_ATTACK_TRAIN}: "
                    "supervised-модель только перекалибруется."
                )

        calib_proba = brute_model.predict_proba(X.iloc[calib_rows][feature_cols])[:, 1]
        quantile = float(np.quantile(calib_proba, 1.0 - args.brute_target_fpr))
        brute_threshold = _strict_threshold(quantile)
        if not 0.0 < brute_threshold < 1.0:
            print(
                f"ВНИМАНИЕ: порог {display_name} вне (0, 1) — benign этой сети не "
                "отделяется от атак; оставлен базовый порог."
            )
            brute_threshold = base_brute.threshold
        adapted_proba = brute_model.predict_proba(X.iloc[holdout_idx][feature_cols])[:, 1]

        # Что на самом деле умеет итоговая модель: при перекалибровке это
        # по-прежнему базовый брутфорс-детектор. Обе модели оцениваются на
        # одних и тех же метках — для брутфорса остальные атаки идут в Other.
        effective = args.classifier if mode == "retrained" else "bruteforce"
        if effective == "bruteforce":
            is_brute = families == BRUTE_FAMILY
            eval_labels = pd.Series(
                np.where(is_brute, BRUTE_FAMILY, np.where(is_benign, "Benign", "Other"))
            )
        else:
            eval_labels = labels
        holdout_view = Dataset(
            name="target", X=X, labels=eval_labels, files=pd.Series(["target"] * len(X))
        )
        # flagged у evaluate_scores — строго «>», у детектора — «>=»: сдвигаем на ulp
        evaluation["supervised"] = {
            "mode": mode,
            "classifier": effective,
            "display_name": display_name,
            "base": _summary(
                evaluate_scores(
                    holdout_view,
                    base_proba,
                    float(np.nextafter(base_brute.threshold, -np.inf)),
                    args.brute_target_fpr,
                    holdout_idx,
                )
            ),
            "adapted": _summary(
                evaluate_scores(
                    holdout_view,
                    adapted_proba,
                    float(np.nextafter(brute_threshold, -np.inf)),
                    args.brute_target_fpr,
                    holdout_idx,
                )
            ),
        }
        brute_payload = {
            **{k: v for k, v in base_brute.metadata.items() if k not in ("metrics", "evaluation")},
            "display_name": display_name,
            "threshold": brute_threshold,
            "target_fpr": float(args.brute_target_fpr),
            "feature_cols": feature_cols,
            "adaptation": {
                "mode": mode,
                "classifier": effective,
                "base_threshold": base_brute.threshold,
                "target_files": [os.path.basename(path) for path in files],
                "positive_train_rows": positive_train,
            },
            "params": rf_params,
            "note": (
                "Адаптировано под целевую сеть: "
                + (
                    f"переобучено на её размеченных данных ({display_name})."
                    if mode == "retrained"
                    else "пересчитан порог по её benign-трафику."
                )
            ),
        }

    # --- Сохранение
    if os.path.exists(output_dir) and args.force:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Сжатие: RF по умолчанию ~109 МБ без него и ~36 МБ с compress=3
    # (больше лимита GitHub на файл); joblib.load распаковывает сам.
    joblib.dump(model, os.path.join(output_dir, "isolation_forest.pkl"), compress=3)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(list(cols), os.path.join(output_dir, "feature_cols_filtered.pkl"))
    _write_json(
        os.path.join(output_dir, STATS_FILE),
        {
            "source": ", ".join(os.path.basename(path) for path in files),
            "rows": int(len(train_rows)),
            "created_at": created_at,
            "feature_cols": list(cols),
            "means": {col: float(value) for col, value in means.reindex(cols).items()},
            "stds": {col: float(value) for col, value in stds.reindex(cols).items()},
            "std_from_base": replaced_std,
        },
    )
    _write_json(
        os.path.join(output_dir, CALIBRATION_FILE),
        {
            "threshold": threshold,
            "target_fpr": float(args.target_fpr),
            "achieved_fpr": {"target holdout": holdout_fpr},
            "rows": int(len(benign_holdout)),
            "quantiles": [
                float(value) for value in np.quantile(holdout_benign_scores, np.linspace(0, 1, 101))
            ],
            "source": "target network holdout",
            "created_at": created_at,
        },
    )
    if brute_payload is not None:
        joblib.dump(brute_model, os.path.join(output_dir, "bruteforce.pkl"), compress=3)
        _write_json(os.path.join(output_dir, "bruteforce.json"), brute_payload)

    metadata = {
        "version": "site-adapted",
        "created_at": created_at,
        "base_model": _display_path(base_dir),
        "target_files": [os.path.basename(path) for path in files],
        "labeled": labeled,
        "split": args.split,
        "holdout": args.holdout,
        "log1p": bool(args.log1p),
        "trim": args.trim,
        "trimmed_rows": trimmed,
        "train_benign_rows": int(len(train_rows)),
        "holdout_rows": int(len(holdout_idx)),
        "holdout_benign_rows": int(len(benign_holdout)),
        "target_fpr": float(args.target_fpr),
        "threshold": threshold,
        "coverage": coverage,
        "unchecked_families": unchecked,
        "evaluation": evaluation,
    }
    _write_json(os.path.join(output_dir, "metadata.json"), metadata)

    checksums = write_checksums(output_dir)
    metadata["signed"] = "signature" in checksums
    return metadata | {"output_dir": output_dir}


def _display_path(path: str) -> str:
    """Путь относительно проекта; на Windows relpath падает между дисками."""
    try:
        relative = os.path.relpath(path, BASE_DIR)
    except ValueError:
        return os.path.abspath(path)
    return os.path.abspath(path) if relative.startswith("..") else relative


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _fmt(value, pattern="{:.3f}") -> str:
    return "—" if value is None else pattern.format(value)


def format_comparison(evaluation: dict) -> str:
    rows = []
    for name, block in evaluation.items():
        if name == "isolation_forest":
            title = "IsolationForest"
        else:
            title = f"{block['display_name']} ({block['mode']})"
        for kind in ("base", "adapted"):
            metrics = block[kind]
            families = ", ".join(
                f"{family} {values['recall']:.2f}"
                for family, values in (metrics.get("families") or {}).items()
                if family != "Other"
            )
            label = f"{title}, {'базовая' if kind == 'base' else 'адаптированная'}"
            rows.append((label, metrics, families))
    width = max(len(label) for label, _, _ in rows)
    lines = [f"{'модель':{width}s} {'FPR':>8s} {'recall':>8s} {'AUC':>7s}  recall по семействам"]
    for label, metrics, families in rows:
        lines.append(
            f"{label:{width}s} {_fmt(metrics['fpr'], '{:.2%}'):>8s} "
            f"{_fmt(metrics['recall']):>8s} {_fmt(metrics['auc']):>7s}  {families}"
        )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    for name in ("target_fpr", "brute_target_fpr", "holdout"):
        if not 0.0 < getattr(args, name) < 1.0:
            print(f"Ошибка: --{name.replace('_', '-')} должен быть в диапазоне (0, 1).")
            return 2
    if not 0.0 <= args.trim < 0.5:
        print("Ошибка: --trim должен быть в диапазоне [0, 0.5).")
        return 2
    if args.rf_trees < 1 or args.rf_depth < 0 or args.rf_leaf < 1:
        print("Ошибка: --rf-trees и --rf-leaf должны быть ≥ 1, --rf-depth ≥ 0.")
        return 2
    if not 0.0 < args.rf_max_samples <= 1.0:
        print("Ошибка: --rf-max-samples должен быть в диапазоне (0, 1].")
        return 2

    try:
        result = adapt(args)
    except (AdaptationError, ModelLoadError, FeatureValidationError, BruteForceError) as exc:
        print(f"Ошибка: {exc}")
        return 2

    print(f"\nПорог IF: score <= {result['threshold']:.4f} (FPR на holdout цели "
          f"{result['evaluation']['isolation_forest']['adapted']['fpr']:.2%})")
    if result["coverage"]:
        print("\nСемейства атак (train / holdout):")
        for family, counts in result["coverage"].items():
            print(f"  {family:14s} {counts['train']:8,} / {counts['holdout']:,}")
    if result["unchecked_families"]:
        print(
            "ВНИМАНИЕ: в holdout нет " + ", ".join(result["unchecked_families"])
            + " — на них модель не проверена. Если строки в файлах не упорядочены "
            "по времени (атаки сгруппированы), используйте --split random."
        )

    print("\nСравнение на отложенном трафике целевой сети:")
    print(format_comparison(result["evaluation"]))
    if not result["labeled"]:
        print("(без меток видны только FPR: recall появится, если добавить колонку Label)")

    output = _display_path(result["output_dir"])
    print(f"\nСохранено в {output}")
    if result["signed"]:
        print(f"{CHECKSUMS_FILE} подписан.")
    elif os.path.isfile(public_key_path()):
        print(
            f"ВНИМАНИЕ: {CHECKSUMS_FILE} не подписан, а публичный ключ есть — приложение "
            f"не загрузит модель. Подпишите: python make_checksums.py --model-dir {output}"
        )
    print(f"Включить: MODEL_DIR={output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
