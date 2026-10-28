"""Кросс-датасетная оценка: обучение на одной сети, проверка на другой.

Метрики на том же датасете, где шло обучение, в CICIDS почти ничего не
говорят: модель выучивает особенности лабораторной сети (ОС, TCP-опции,
версию CICFlowMeter), а не атаки. Здесь для каждой упорядоченной пары
датасетов (A -> B) модель обучается и калибруется только на A и
оценивается на B. Для сравнения считается и in-domain (A -> A) на
случайном holdout — это оптимистичная верхняя граница.

Детекторы:
- if  — IsolationForest на benign A, порог по квантилю benign holdout A;
- rf  — RandomForest attack/benign на A, порог по квантилю benign holdout A;
- hgb — HistGradientBoosting attack/benign на A, порог так же.

Варианты признаков:
- baseline  — все признаки модели;
- drop_net  — без признаков, зависящих от сети, а не от атаки
  (длины заголовков, начальные окна TCP, минимальный размер сегмента);
- drop_time — без длительностей, IAT, скоростей и Active/Idle;
- log1p     — sign(x)·log1p(|x|) перед StandardScaler;
- domain    — log1p и нормировка (x - медиана) / IQR по benign той сети,
  где модель применяется; порог тоже ставится по этому benign. Так
  выглядело бы развёртывание: на новой сети снимают эталонный нормальный
  трафик (метки атак не нужны). Для честности эти benign-строки цели
  исключаются из оценки у всех вариантов.
log1p монотонен, поэтому для деревьев (rf, hgb) ничего не меняет: для них
log1p без domain не считается.

Смесь сетей (--mix a+b) — отдельный источник обучения; её проверяют только
на сетях, которых в смеси нет.

Пример (датасеты dhoogla из кэша kagglehub находятся автоматически):
    python evaluate.py --output evaluation.json
    python evaluate.py --mix cic2017+cic2018 --targets ddos2019
    python evaluate.py --dataset cic2017=data/2017 --dataset cic2018=data/2018
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.model_loader import load_model_objects
from modules.pipeline import prepare_features

KAGGLE_DATASETS = {
    "cic2017": "dhoogla/cicids2017",
    "cic2018": "dhoogla/csecicids2018",
    "ddos2019": "dhoogla/cicddos2019",
}

# Признаки, которые описывают окружение (ОС, TCP-опции, стек), а не атаку:
# у SSH-брутфорса Bwd Header Length 1032 в CIC-IDS2017 и 712 в 2018 при
# почти одинаковых длинах пакетов.
NETWORK_FEATURES = [
    "Fwd Header Length",
    "Bwd Header Length",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Fwd Seg Size Min",
]

# Временные признаки: длительности, IAT, скорости, Active/Idle. Между
# датасетами они расходятся сильнее всего (медиана Fwd IAT Total 48 мкс в
# 2017 против 184 мс в 2018) — похоже, из-за разных настроек CICFlowMeter.
TIME_FEATURE_RE = re.compile(r"IAT|Duration|/s$|^Active |^Idle ")

VARIANTS = {
    "baseline": {},
    "drop_net": {"drop_net": True},
    "drop_time": {"drop_time": True},
    "log1p": {"log": True},
    "drop_net+log1p": {"drop_net": True, "log": True},
    # Нормализация под сеть: log1p, затем (x - медиана) / IQR по benign
    # той сети, где модель применяется, и порог по тому же benign.
    "domain": {"domain": True},
    "domain+drop_time": {"domain": True, "drop_time": True},
}
DEFAULT_VARIANTS = ["baseline", "drop_time", "log1p", "domain", "domain+drop_time"]
DETECTORS = ("if", "rf", "hgb")
SUPERVISED = ("rf", "hgb")

# Порядок важен: "Brute Force -Web" — это Web, "DDoS" — не DoS. Метки
# CIC-DDoS2019 (DrDoS_DNS, MSSQL, Syn, UDP-lag...) — все DDoS; проверяются
# первыми и целиком, чтобы MSSQL не попал в Web, а DrDoS — в DoS.
_FAMILIES = [
    (
        "DDoS",
        re.compile(
            r"^(drdos_\w+|ldap|netbios|mssql|portmap|syn|tftp|udp|udp-?lag|webddos)$", re.I
        ),
    ),
    ("Web", re.compile(r"web|xss|sql", re.I)),
    ("DDoS", re.compile(r"ddos", re.I)),
    ("DoS", re.compile(r"dos|heartbleed", re.I)),
    ("BruteForce", re.compile(r"patator|brute", re.I)),
    ("Bot", re.compile(r"\bbot\b", re.I)),
    ("Infiltration", re.compile(r"infil", re.I)),
    ("PortScan", re.compile(r"portscan", re.I)),
]


def attack_family(label: str) -> str:
    """Сводит метки разных датасетов к общим семействам атак."""
    text = str(label).strip()
    if text.lower() == "benign":
        return "Benign"
    for family, pattern in _FAMILIES:
        if pattern.search(text):
            return family
    return "Other"


def transform(
    X: pd.DataFrame, drop_net: bool = False, drop_time: bool = False, log: bool = False
) -> pd.DataFrame:
    if drop_net:
        X = X.drop(columns=[c for c in NETWORK_FEATURES if c in X.columns])
    if drop_time:
        X = X.drop(columns=[c for c in X.columns if TIME_FEATURE_RE.search(c)])
    if log:
        # Знаковый log1p: в признаках бывают отрицательные значения
        # (Init Win Bytes = -1, отрицательные IAT из-за рассинхрона часов).
        X = np.sign(X) * np.log1p(np.abs(X))
    return X


def robust_stats(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Медиана и масштаб (IQR, иначе std, иначе 1) по benign одной сети."""
    median = X.median()
    scale = X.quantile(0.75) - X.quantile(0.25)
    std = X.std()
    scale = scale.where(scale > 0, std).where(lambda s: s > 0, 1.0)
    return median, scale


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    labels: pd.Series
    files: pd.Series

    @property
    def y(self) -> np.ndarray:
        return (self.labels.str.lower() != "benign").to_numpy()


def _kaggle_path(slug: str) -> str | None:
    root = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets", *slug.split("/")
    )
    versions = [
        path
        for path in glob.glob(os.path.join(root, "versions", "*"))
        if os.path.basename(path).isdigit()
    ]
    if not versions:
        return None
    return max(versions, key=lambda path: int(os.path.basename(path)))


def resolve_datasets(specs: list[str] | None) -> dict[str, str]:
    if specs:
        result = {}
        for spec in specs:
            name, sep, path = spec.partition("=")
            if not sep or not name or not path:
                raise ValueError(f"--dataset ожидает NAME=PATH, получено: {spec!r}")
            result[name] = path
        return result

    result = {}
    for name, slug in KAGGLE_DATASETS.items():
        path = _kaggle_path(slug)
        if path is not None:
            result[name] = path
    if len(result) < 2:
        raise ValueError(
            "В кэше kagglehub меньше двух датасетов из: "
            + ", ".join(KAGGLE_DATASETS.values())
            + ". Скачайте их (kagglehub.dataset_download(...)) или передайте --dataset NAME=PATH."
        )
    return result


def augment_dataset(base: Dataset, donor: Dataset, families: list[str]) -> Dataset:
    """base плюс только атаки указанных семейств из donor (без его benign)."""
    mask = donor.labels.map(attack_family).isin(families).to_numpy()
    if not mask.any():
        raise ValueError(f"В {donor.name} нет атак семейств: {', '.join(families)}")
    part = Dataset(
        name=donor.name,
        X=donor.X[mask].reset_index(drop=True),
        labels=donor.labels[mask].reset_index(drop=True),
        files=donor.files[mask].reset_index(drop=True),
    )
    combined = combine_datasets([base, part])
    combined.name = f"{base.name}+{donor.name}[{','.join(families)}]"
    return combined


def combine_datasets(datasets: list[Dataset]) -> Dataset:
    """Смесь сетей как один источник обучения (имя: a+b)."""
    return Dataset(
        name="+".join(d.name for d in datasets),
        X=pd.concat([d.X for d in datasets], ignore_index=True),
        labels=pd.concat([d.labels for d in datasets], ignore_index=True),
        files=pd.concat([d.name + "/" + d.files for d in datasets], ignore_index=True),
    )


def load_dataset(
    name: str,
    path: str,
    cols: list[str],
    max_benign: int,
    max_attack: int,
    seed: int,
) -> Dataset:
    """Стратифицированная выборка: до max_benign benign и до max_attack строк
    каждой атаки из каждого файла, чтобы редкие классы не терялись."""
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise ValueError(f"В {path} нет .parquet файлов.")

    rng = np.random.default_rng(seed)
    parts = []
    for file in files:
        df = pd.read_parquet(file, columns=cols + ["Label"])
        df["Label"] = df["Label"].astype(str).str.strip()
        for label, group in df.groupby("Label", sort=False):
            limit = max_benign if label.lower() == "benign" else max_attack
            if limit and len(group) > limit:
                group = group.iloc[rng.choice(len(group), limit, replace=False)]
            part = group.copy()
            part["_file"] = os.path.basename(file)
            parts.append(part)

    data = pd.concat(parts, ignore_index=True)
    X = data[cols].replace([np.inf, -np.inf], np.nan).astype("float64")
    return Dataset(name=name, X=X, labels=data["Label"], files=data["_file"])


def _split(n: int, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1.0 - holdout))
    return order[:cut], order[cut:]


class Detector:
    """Обучение на источнике + порог по benign holdout источника.

    В варианте domain признаки дополнительно нормализуются по benign той
    сети, где модель применяется (adapt), — как при развёртывании, когда
    на новой сети снимают эталонный нормальный трафик.
    """

    def __init__(self, kind: str, variant: dict, target_fpr: float, seed: int, n_jobs: int):
        self.kind = kind
        self.variant = variant
        self.domain = bool(variant.get("domain"))
        self.transform_args = {
            "drop_net": bool(variant.get("drop_net")),
            "drop_time": bool(variant.get("drop_time")),
            "log": bool(variant.get("log")) or self.domain,
        }
        self.target_fpr = target_fpr
        self.seed = seed
        self.n_jobs = n_jobs
        self.norm = None

    def _raw_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Пропуски заполняются средними benign источника — как в pipeline
        X, _ = prepare_features(X, list(X.columns), fill_values=self.fill_values)
        return transform(X, **self.transform_args)

    def _features(self, X: pd.DataFrame, norm=None) -> pd.DataFrame:
        features = self._raw_features(X)
        norm = norm or self.norm
        if self.domain and norm is not None:
            median, scale = norm
            features = (features - median) / scale
        return features

    def adapt(self, benign: pd.DataFrame) -> tuple[tuple[pd.Series, pd.Series] | None, float]:
        """Нормировка и порог по benign целевой сети (метки атак не нужны)."""
        norm = robust_stats(self._raw_features(benign)) if self.domain else None
        scores = self.score(benign, norm)
        return norm, float(np.quantile(scores, 1.0 - self.target_fpr))

    def fit(self, source: Dataset, holdout: float, max_train: int) -> np.ndarray:
        """Обучает модель; возвращает индексы in-domain holdout источника."""
        y = source.y
        train_idx, holdout_idx = _split(len(source.X), holdout, self.seed)
        benign_train = train_idx[~y[train_idx]]
        self.fill_values = source.X.iloc[benign_train].mean()

        if self.kind == "if":
            idx = benign_train
        else:
            idx = train_idx
        if max_train and len(idx) > max_train:
            idx = np.random.default_rng(self.seed).choice(idx, max_train, replace=False)

        if self.domain:
            self.norm = robust_stats(self._raw_features(source.X.iloc[benign_train]))
        X_train = self._features(source.X.iloc[idx])
        self.columns = list(X_train.columns)

        if self.kind == "if":
            self.scaler = StandardScaler().fit(X_train)
            self.model = IsolationForest(
                n_estimators=100, random_state=self.seed, n_jobs=self.n_jobs
            ).fit(self.scaler.transform(X_train))
        elif self.kind == "rf":
            self.scaler = None
            self.model = RandomForestClassifier(
                n_estimators=60,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=self.seed,
                n_jobs=self.n_jobs,
            ).fit(X_train, y[idx])
        else:
            self.scaler = None
            self.model = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=self.seed,
            ).fit(X_train, y[idx])

        benign_holdout = holdout_idx[~y[holdout_idx]]
        holdout_scores = self.score(source.X.iloc[benign_holdout])
        self.threshold = float(np.quantile(holdout_scores, 1.0 - self.target_fpr))
        return holdout_idx

    def score(self, X: pd.DataFrame, norm=None) -> np.ndarray:
        """Чем больше, тем аномальнее (у IF знак decision_function инвертирован)."""
        features = self._features(X, norm)
        if self.kind == "if":
            return -self.model.decision_function(self.scaler.transform(features))
        return self.model.predict_proba(features)[:, 1]


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    if y.all() or not y.any():
        return None
    return float(roc_auc_score(y, scores))


def evaluate_scores(
    target: Dataset,
    scores: np.ndarray,
    threshold: float,
    target_fpr: float,
    index: np.ndarray | None = None,
) -> dict:
    """Метрики на целевом датасете.

    recall@src — с порогом, откалиброванным на источнике (как в проде);
    recall@oracle — с порогом по benign цели на тот же FPR: отделяет
    качество ранжирования от сдвига калибровки.
    """
    if index is None:
        index = np.arange(len(target.X))
    y = target.y[index]
    families = target.labels.iloc[index].map(attack_family).to_numpy()
    # Строго «>»: у RF много одинаковых вероятностей (часто ровно 0), и при
    # «>=» квантиль 0 пометил бы весь трафик. Так FPR не выше целевого.
    flagged = scores > threshold

    benign_scores = scores[~y]
    oracle = float(np.quantile(benign_scores, 1.0 - target_fpr)) if len(benign_scores) else threshold
    flagged_oracle = scores > oracle

    result = {
        "rows": int(len(y)),
        "attacks": int(y.sum()),
        "auc": _safe_auc(y, scores),
        "fpr": float(flagged[~y].mean()) if (~y).any() else None,
        "recall": float(flagged[y].mean()) if y.any() else None,
        "recall_oracle": float(flagged_oracle[y].mean()) if y.any() else None,
        "families": {},
    }
    for family in sorted(set(families) - {"Benign"}):
        mask = families == family
        subset = mask | ~y
        result["families"][family] = {
            "n": int(mask.sum()),
            "auc": _safe_auc(mask[subset], scores[subset]),
            "recall": float(flagged[mask].mean()),
            "recall_oracle": float(flagged_oracle[mask].mean()),
        }
    return result


def run(
    datasets: dict[str, Dataset],
    detectors: list[str],
    variants: list[str],
    target_fpr: dict[str, float],
    holdout: float,
    max_train: int,
    seed: int,
    n_jobs: int,
    adapt_share: float = 0.2,
    mixes: list[list[str]] | tuple = (),
    sources: list[str] | None = None,
    targets: list[str] | None = None,
    augments: list[tuple[str, str, list[str]]] | tuple = (),
    log=print,
) -> list[dict]:
    """Все пары источник -> цель; цель никогда не входит в состав источника.

    mixes — смеси сетей как дополнительные источники (["cic2017", "cic2018"]);
    augments — (base, donor, families): base плюс только атаки этих семейств
    из donor (имя "cic2017+cic2018[DDoS]"). На donor такой источник
    проверяется без этих семейств — они были в обучении;
    sources/targets — ограничить имена.
    """
    components = {name: {name} for name in datasets}
    excluded: dict[str, dict[str, set[str]]] = {}
    all_sources = dict(datasets)
    for mix in mixes:
        combined = combine_datasets([datasets[name] for name in mix])
        all_sources[combined.name] = combined
        components[combined.name] = set(mix)
    for base, donor, families in augments:
        augmented = augment_dataset(datasets[base], datasets[donor], families)
        all_sources[augmented.name] = augmented
        components[augmented.name] = {base}
        excluded[augmented.name] = {donor: set(families)}
    if sources:
        all_sources = {name: all_sources[name] for name in sources}
    target_names = list(targets) if targets else list(datasets)

    # Из benign каждой цели заранее откладывается adapt_share строк — по ним
    # domain-варианты нормируют признаки и ставят порог. Оценка у всех
    # вариантов идёт на остальных строках, чтобы цифры были сопоставимы.
    splits = {}
    for name, data in datasets.items():
        benign = np.flatnonzero(~data.y)
        rng = np.random.default_rng(seed + 1)
        adapt_idx = rng.choice(benign, int(len(benign) * adapt_share), replace=False)
        eval_mask = np.ones(len(data.X), dtype=bool)
        eval_mask[adapt_idx] = False
        splits[name] = (adapt_idx, np.flatnonzero(eval_mask))

    results = []
    for source_name, source in all_sources.items():
        route_targets = [source_name] + [
            name for name in target_names if name not in components[source_name]
        ]
        for kind in detectors:
            for variant_name in variants:
                variant = VARIANTS[variant_name]
                if kind in SUPERVISED and variant.get("log") and not variant.get("domain"):
                    continue  # монотонное преобразование не меняет деревья
                started = time.time()
                detector = Detector(kind, variant, target_fpr[kind], seed, n_jobs)
                holdout_idx = detector.fit(source, holdout, max_train)
                for target_name in route_targets:
                    target = datasets.get(target_name)
                    threshold_from = "source"
                    skip = None
                    if target_name == source_name:
                        scores = detector.score(source.X.iloc[holdout_idx])
                        metrics = evaluate_scores(
                            source, scores, detector.threshold, target_fpr[kind], holdout_idx
                        )
                    else:
                        adapt_idx, eval_idx = splits[target_name]
                        skip = excluded.get(source_name, {}).get(target_name)
                        if skip:
                            families = target.labels.iloc[eval_idx].map(attack_family)
                            eval_idx = eval_idx[~families.isin(skip).to_numpy()]
                        norm, threshold = None, detector.threshold
                        if detector.domain:
                            norm, threshold = detector.adapt(target.X.iloc[adapt_idx])
                            threshold_from = "target benign"
                        scores = detector.score(target.X.iloc[eval_idx], norm)
                        metrics = evaluate_scores(
                            target, scores, threshold, target_fpr[kind], eval_idx
                        )
                    results.append(
                        {
                            "source": source_name,
                            "target": target_name,
                            "in_domain": target_name == source_name,
                            "detector": kind,
                            "variant": variant_name,
                            "features": len(detector.columns),
                            "target_fpr": target_fpr[kind],
                            "threshold_from": threshold_from,
                            "excluded_families": sorted(skip) if skip else [],
                            **metrics,
                        }
                    )
                log(
                    f"  {source_name} {kind:2s} {variant_name:15s} "
                    f"{time.time() - started:5.1f}s"
                )
    return results


def _fmt(value, pattern="{:.3f}") -> str:
    return "  —  " if value is None else pattern.format(value)


def format_report(results: list[dict]) -> str:
    lines = []
    families = sorted({f for r in results for f in r["families"]})
    def route(r: dict) -> str:
        text = f"{r['source']}→{r['target']}"
        if r["in_domain"]:
            text += " (in)"
        if r.get("excluded_families"):
            text += f" без {','.join(r['excluded_families'])}"
        return text

    width = max(len(route(r)) for r in results)
    header = (
        f"{'source→target':{width}s} {'det':3s} {'variant':16s} {'AUC':>6s} {'FPR':>7s} "
        f"{'recall':>7s} {'r@orcl':>7s} | "
        + " ".join(f"{f[:8]:>8s}" for f in families)
    )
    lines.append("Семейства: AUC против всего benign цели")
    lines.append(header)
    lines.append("-" * len(header))
    # Источники одной цели — рядом, чтобы сравнивать, на чём лучше учить
    ordered = sorted(
        results,
        key=lambda r: (
            r["in_domain"],
            r["target"],
            r["detector"],
            list(VARIANTS).index(r["variant"]),
            r["source"].count("+"),
            r["source"],
        ),
    )
    for r in ordered:
        cells = " ".join(
            f"{_fmt(r['families'].get(f, {}).get('auc')):>8s}" for f in families
        )
        lines.append(
            f"{route(r):{width}s} {r['detector']:3s} {r['variant']:16s} {_fmt(r['auc']):>6s} "
            f"{_fmt(r['fpr'], '{:.2%}'):>7s} {_fmt(r['recall']):>7s} "
            f"{_fmt(r['recall_oracle']):>7s} | {cells}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Кросс-датасетная оценка детекторов")
    parser.add_argument(
        "--dataset",
        action="append",
        help="NAME=PATH к каталогу с parquet (повторяемый). По умолчанию — "
        "cic2017, cic2018 и ddos2019, какие найдутся в кэше kagglehub.",
    )
    parser.add_argument(
        "--mix",
        action="append",
        default=[],
        help="Смесь сетей как источник обучения, например cic2017+cic2018 (повторяемый)",
    )
    parser.add_argument(
        "--augment",
        action="append",
        default=[],
        help="BASE+DONOR:FAMILY[,FAMILY] — BASE плюс только атаки этих семейств из "
        "DONOR, например cic2017+cic2018:DDoS (повторяемый)",
    )
    parser.add_argument("--sources", nargs="+", default=None, help="Ограничить источники")
    parser.add_argument("--targets", nargs="+", default=None, help="Ограничить цели")
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=DETECTORS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    parser.add_argument(
        "--adapt-share",
        type=float,
        default=0.2,
        help="Доля benign цели для нормировки/порога в domain-вариантах",
    )
    parser.add_argument("--max-benign-per-file", type=int, default=100_000)
    parser.add_argument("--max-attack-per-label", type=int, default=50_000)
    parser.add_argument("--max-train", type=int, default=400_000, help="Лимит строк на обучение")
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--if-target-fpr", type=float, default=0.02)
    parser.add_argument("--rf-target-fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", default=None, help="Куда сохранить JSON с метриками")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    for name in ("holdout", "adapt_share", "if_target_fpr", "rf_target_fpr"):
        if not 0.0 < getattr(args, name) < 1.0:
            print(f"Ошибка: --{name.replace('_', '-')} должен быть в диапазоне (0, 1).")
            return 2

    try:
        paths = resolve_datasets(args.dataset)
    except ValueError as exc:
        print(f"Ошибка: {exc}")
        return 2
    if len(paths) < 2:
        print("Ошибка: нужно минимум два датасета для оценки переноса.")
        return 2

    mixes = [spec.split("+") for spec in args.mix]
    for mix in mixes:
        unknown = [name for name in mix if name not in paths]
        if len(mix) < 2 or unknown:
            print(f"Ошибка: --mix {'+'.join(mix)}: нужно ≥2 известных датасета "
                  f"(есть: {', '.join(paths)}).")
            return 2
    augments = []
    for spec in args.augment:
        names, sep, family_text = spec.partition(":")
        base, plus, donor = names.partition("+")
        families = [f.strip() for f in family_text.split(",") if f.strip()]
        if not (sep and plus and families) or base not in paths or donor not in paths:
            print(f"Ошибка: --augment {spec}: ожидается BASE+DONOR:FAMILY из {', '.join(paths)}.")
            return 2
        augments.append((base, donor, families))
    known_sources = (
        set(paths)
        | {"+".join(mix) for mix in mixes}
        | {f"{b}+{d}[{','.join(f)}]" for b, d, f in augments}
    )
    for option, names, known in (
        ("--sources", args.sources, known_sources),
        ("--targets", args.targets, set(paths)),
    ):
        unknown = [name for name in names or [] if name not in known]
        if unknown:
            print(f"Ошибка: {option}: неизвестные имена {', '.join(unknown)}.")
            return 2

    # Признаки — те же, что у рабочей модели, чтобы выводы к ней применимы
    _, _, _, cols = load_model_objects()

    datasets = {}
    for name, path in paths.items():
        try:
            datasets[name] = load_dataset(
                name, path, cols, args.max_benign_per_file, args.max_attack_per_label, args.seed
            )
        except (OSError, ValueError, KeyError) as exc:
            print(f"Ошибка загрузки {name} ({path}): {exc}")
            return 2
        data = datasets[name]
        print(f"{name}: {len(data.X):,} строк, атак {int(data.y.sum()):,} ({path})")

    print("\nОбучение:")
    results = run(
        datasets,
        args.detectors,
        args.variants,
        {"if": args.if_target_fpr, "rf": args.rf_target_fpr, "hgb": args.rf_target_fpr},
        args.holdout,
        args.max_train,
        args.seed,
        args.n_jobs,
        adapt_share=args.adapt_share,
        mixes=mixes,
        sources=args.sources,
        targets=args.targets,
        augments=augments,
    )

    print()
    print(format_report(results))
    print(
        "\nrecall — с порогом источника (у domain — с порогом по отложенному benign цели);\n"
        "r@orcl — с порогом по оцениваемому benign цели на тот же FPR.\n"
        "(in) — in-domain на случайном holdout: оптимистичная оценка."
    )

    if args.output:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "datasets": paths,
            "network_features": NETWORK_FEATURES,
            "params": {
                k: v
                for k, v in vars(args).items()
                if k not in ("dataset", "output", "mix", "augment")
            },
            "mixes": ["+".join(mix) for mix in mixes],
            "augments": args.augment,
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nСохранено: {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
