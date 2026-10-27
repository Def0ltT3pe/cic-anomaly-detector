"""Загрузка модели и статистик нормального трафика.

Все пути строятся от каталога проекта, поэтому приложение можно запускать
из любого рабочего каталога: локально, из IDE, из Docker.

Статистики хранятся в JSON (`models/normal_stats.json`), а не в pickle,
чтобы файл не зависел от версии pandas и читался без исполнения кода.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from modules.integrity import IntegrityError, verify_checksums, verify_if_listed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "model": "isolation_forest.pkl",
    "scaler": "scaler.pkl",
    "selector": "selector.pkl",
    "feature_cols": "feature_cols_filtered.pkl",
}
STATS_FILE = "normal_stats.json"
TRAIN_PARQUET = "Benign-Monday-no-metadata.parquet"


class ModelLoadError(RuntimeError):
    """Модель или статистики не найдены либо повреждены."""


def resolve_model_dir(model_dir: str | None = None) -> str:
    """Каталог моделей: аргумент -> переменная MODEL_DIR -> models/."""
    if not model_dir:
        model_dir = os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(BASE_DIR, model_dir)
    return os.path.abspath(model_dir)


def load_model_objects(model_dir: str | None = None):
    """Возвращает (model, scaler, selector, feature_cols)."""
    model_dir = resolve_model_dir(model_dir)
    paths = {key: os.path.join(model_dir, name) for key, name in MODEL_FILES.items()}

    required = ["model", "scaler", "feature_cols"]
    missing = [paths[key] for key in required if not os.path.isfile(paths[key])]
    if missing:
        raise ModelLoadError(
            "Не найдены файлы модели: "
            + ", ".join(missing)
            + f". Ожидаемый каталог: {model_dir}"
        )

    try:
        artifacts = [paths["model"], paths["scaler"], paths["feature_cols"]]
        if os.path.isfile(paths["selector"]):
            artifacts.append(paths["selector"])
        verify_checksums(model_dir, artifacts)
    except IntegrityError as exc:
        raise ModelLoadError(str(exc)) from exc

    try:
        model = joblib.load(paths["model"])
        scaler = joblib.load(paths["scaler"])
        feature_cols = list(joblib.load(paths["feature_cols"]))
        # Селектор необязателен: в v2 он не используется (см. README).
        selector = (
            joblib.load(paths["selector"])
            if os.path.isfile(paths["selector"])
            else None
        )
    except Exception as exc:  # повреждённый/несовместимый pickle
        raise ModelLoadError(f"Не удалось прочитать модель из {model_dir}: {exc}") from exc

    if not feature_cols:
        raise ModelLoadError(f"Пустой список признаков в {paths['feature_cols']}")

    return model, scaler, selector, feature_cols


def _find_training_parquet(model_dir: str, parquet_path: str | None) -> str | None:
    candidates = []
    if parquet_path:
        candidates.append(parquet_path)
    candidates += [
        os.path.join(model_dir, TRAIN_PARQUET),
        os.path.join(BASE_DIR, TRAIN_PARQUET),
        os.path.join(BASE_DIR, "data", TRAIN_PARQUET),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def _compute_normal_stats(parquet_path: str, feature_cols: list[str]):
    """Считает среднее/стандартное отклонение по benign-трафику."""
    df = pd.read_parquet(parquet_path)
    if "Label" not in df.columns:
        raise ModelLoadError(f"В обучающем файле {parquet_path} нет колонки Label.")

    constant = [c for c in df.columns if c != "Label" and df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)

    labels = df["Label"].astype(str).str.strip().str.lower()
    benign = df[labels == "benign"]
    if benign.empty:
        raise ModelLoadError(
            f"В {os.path.basename(parquet_path)} не найдено строк с меткой Benign."
        )

    missing = [c for c in feature_cols if c not in benign.columns]
    if missing:
        raise ModelLoadError(
            "В обучающем файле отсутствуют признаки: " + ", ".join(missing)
        )

    means = benign[feature_cols].mean()
    stds = benign[feature_cols].std()

    if not np.isfinite(means.to_numpy()).all() or not np.isfinite(stds.to_numpy()).all():
        raise ModelLoadError("Статистики содержат NaN/inf — проверьте обучающие данные.")

    zero = stds[stds <= 0]
    if len(zero):
        raise ModelLoadError(
            "Нулевое стандартное отклонение у признаков: " + ", ".join(zero.index)
        )

    return means.astype(float), stds.astype(float), int(len(benign))


def _read_stats_file(stats_path: str, feature_cols: list[str]):
    with open(stats_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    means = pd.Series(payload["means"], dtype=float)
    stds = pd.Series(payload["stds"], dtype=float)

    missing = [c for c in feature_cols if c not in means.index or c not in stds.index]
    if missing:
        raise ModelLoadError(
            f"В {os.path.basename(stats_path)} нет статистик для признаков: "
            + ", ".join(missing)
        )

    means = means.reindex(feature_cols)
    stds = stds.reindex(feature_cols)

    if not np.isfinite(means.to_numpy()).all() or not np.isfinite(stds.to_numpy()).all():
        raise ModelLoadError(f"Файл {stats_path} содержит NaN/inf в статистиках.")
    if (stds <= 0).any():
        raise ModelLoadError(f"Файл {stats_path} содержит нулевые стандартные отклонения.")

    return means, stds


def load_normal_stats(
    feature_cols: list[str],
    model_dir: str | None = None,
    parquet_path: str | None = None,
    rebuild: bool = False,
):
    """Возвращает (means, stds) для benign-трафика.

    Порядок: готовый normal_stats.json -> расчёт из parquet (и сохранение).
    """
    model_dir = resolve_model_dir(model_dir)
    stats_path = os.path.join(model_dir, STATS_FILE)

    if os.path.isfile(stats_path) and not rebuild:
        try:
            # Средние идут в заполнение пропусков, поэтому подмена файла
            # меняет признаки на входе модели — проверяем, если он подписан.
            verify_if_listed(model_dir, [STATS_FILE])
        except IntegrityError as exc:
            raise ModelLoadError(str(exc)) from exc
        try:
            return _read_stats_file(stats_path, feature_cols)
        except (KeyError, ValueError, TypeError) as exc:
            raise ModelLoadError(
                f"Не удалось прочитать {stats_path}: {exc}. "
                "Удалите файл или запустите с rebuild=True."
            ) from exc

    parquet = _find_training_parquet(model_dir, parquet_path)
    if parquet is None:
        raise ModelLoadError(
            f"Нет {STATS_FILE} и не найден {TRAIN_PARQUET}. "
            "Положите normal_stats.json в models/ или тренировочный parquet "
            "в корень проекта/каталог data/."
        )

    means, stds, rows = _compute_normal_stats(parquet, feature_cols)

    payload = {
        "source": os.path.basename(parquet),
        "rows": rows,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_cols": list(feature_cols),
        "means": {col: float(value) for col, value in means.items()},
        "stds": {col: float(value) for col, value in stds.items()},
    }
    try:
        os.makedirs(model_dir, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise ModelLoadError(
            f"Не удалось сохранить {stats_path}: {exc}. "
            "Проверьте права на каталог models/."
        ) from exc

    return means, stds
