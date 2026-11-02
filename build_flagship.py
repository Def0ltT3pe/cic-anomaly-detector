"""Флагманская модель: одна IsolationForest и один классификатор атак на всех сетях.

Сети: CIC-IDS2017 и CSE-CIC-IDS2018 (датасеты dhoogla из кэша kagglehub) и
свои серверы (размеченные потоки: pcap_to_flows.py + label_from_authlog.py).
Каждая сеть входит в обучение и нормой, и атаками, с одинаковыми лимитами —
иначе модель выучивает «эта сеть = атака» (так вышло со смесью 2017+2018:
FPR на benign 2018 вырос с 3.9% до 19.6%).

Проверка — честная:
- «без одной сети»: учим на остальных, проверяем на отложенной части
  выброшенной сети — так видно, как модель ведёт себя на незнакомой сети;
- CIC-DDoS2019 — сеть, которой нет ни в одном обучении;
- итоговая модель на отложенной части каждой сети против базовой модели
  (models/) и, если указаны, моделей, адаптированных под серверы.

Отложенная часть: у CIC — случайные 20% (атаки в файлах сгруппированы),
у серверов — последние файлы по времени (имя capture_NNNNN_…).

Пример:
    python build_flagship.py \\
        --site serverA=D:/flows/serverA:3 --site serverB=D:/flows/serverB:11 \\
        --compare serverA=D:/models/serverA --compare serverB=D:/models/serverB \\
        --output-dir models_flagship
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluate import (
    SUMMARY_KEYS,
    Dataset,
    attack_family,
    evaluate_scores,
    load_dataset,
    resolve_datasets,
    summarize_metrics,
)
from modules.bruteforce import load_bruteforce
from modules.calibration import CALIBRATION_FILE, load_calibration
from modules.integrity import write_checksums, write_json
from modules.model_loader import STATS_FILE, load_model_objects, load_normal_stats, resolve_model_dir
from modules.pipeline import if_scores, make_scaler, prepare_features

TRAIN_NETWORKS = ("cic2017", "cic2018")
EXTERNAL_NETWORK = "ddos2019"
DISPLAY_NAME = "Атаки"
# recall_oracle показывает, хватает ли ранжирования, если порог подобрать под сеть
REPORT_KEYS = SUMMARY_KEYS + ("recall_oracle",)


class FlagshipError(RuntimeError):
    pass


@dataclass
class Network:
    """Выборка одной сети: train (обучение), calib (порог), test (проверка)."""

    name: str
    data: Dataset
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Флагманская модель на всех сетях")
    parser.add_argument(
        "--site", action="append", default=[],
        help="NAME=DIR[:N] — размеченные потоки сервера (*.parquet с Label); "
        "последние N файлов — проверка (по умолчанию 15%%)",
    )
    parser.add_argument(
        "--compare", action="append", default=[],
        help="NAME=MODEL_DIR — модель, адаптированная под сеть NAME, для сравнения",
    )
    parser.add_argument("--output-dir", default="models_flagship")
    parser.add_argument("--force", action="store_true", help="Перезаписать --output-dir")
    parser.add_argument("--max-benign", type=int, default=80_000, help="Норма на сеть")
    parser.add_argument("--max-attack", type=int, default=40_000, help="Атак на сеть")
    parser.add_argument("--max-external", type=int, default=40_000,
                        help="Строк CIC-DDoS2019 на каждый класс (норма / атаки)")
    parser.add_argument("--target-fpr", type=float, default=0.02, help="FPR IsolationForest")
    parser.add_argument("--rf-target-fpr", type=float, default=0.001, help="FPR классификатора")
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--rf-depth", type=int, default=20)
    parser.add_argument("--rf-leaf", type=int, default=5)
    parser.add_argument("--rf-max-samples", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=200, help="Деревьев IsolationForest")
    parser.add_argument("--no-lono", action="store_true", help="Без проверки «без одной сети»")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def _cap(labels: pd.Series, max_benign: int, max_attack: int, rng) -> np.ndarray:
    """Индексы: до max_benign нормы и до max_attack атак, атаки — поровну по семействам."""
    benign = np.flatnonzero((labels.str.lower() == "benign").to_numpy())
    keep = [benign if len(benign) <= max_benign else rng.choice(benign, max_benign, replace=False)]
    families = labels.map(attack_family).to_numpy()
    attack_families = sorted(set(families) - {"Benign"})
    budget = max_attack
    # Малые семейства берём целиком, остаток бюджета делим между крупными
    for position, family in enumerate(sorted(attack_families, key=lambda f: (families == f).sum())):
        rows = np.flatnonzero(families == family)
        share = budget // (len(attack_families) - position)
        if len(rows) > share:
            rows = rng.choice(rows, share, replace=False)
        keep.append(rows)
        budget -= len(rows)
    return np.sort(np.concatenate(keep))


def _subset(data: Dataset, rows: np.ndarray) -> Dataset:
    return Dataset(
        name=data.name,
        X=data.X.iloc[rows].reset_index(drop=True),
        labels=data.labels.iloc[rows].reset_index(drop=True),
        files=data.files.iloc[rows].reset_index(drop=True),
    )


def _split_calib(data: Dataset, train: np.ndarray, rng, share: float = 0.2):
    """Из обучающей части — 20% нормы на порог (модель её не видит)."""
    benign = train[~data.y[train]]
    calib = rng.choice(benign, int(len(benign) * share), replace=False)
    return np.setdiff1d(train, calib), np.sort(calib)


def load_cic(name: str, path: str, cols: list[str], args, rng) -> Network:
    raw = load_dataset(name, path, cols, args.max_benign, args.max_attack, args.seed)
    data = _subset(raw, _cap(raw.labels, args.max_benign, args.max_attack, rng))
    order = rng.permutation(len(data.X))
    cut = int(len(order) * 0.8)
    train, calib = _split_calib(data, np.sort(order[:cut]), rng)
    return Network(name, data, train, calib, np.sort(order[cut:]))


def parse_site_spec(spec: str) -> tuple[str, str, int | None]:
    """NAME=DIR[:N] -> (имя, каталог, N). Двоеточие диска Windows (D:/…) — не N."""
    name, sep, rest = spec.partition("=")
    directory, count = rest, None
    head, colon, tail = rest.rpartition(":")
    if colon and tail.isdigit():
        directory, count = head, int(tail)
    if not sep or not name or not directory:
        raise FlagshipError(f"--site ожидает NAME=DIR[:N], получено: {spec!r}")
    return name, directory, count


def load_site(spec: str, cols: list[str], args, rng) -> Network:
    name, directory, count = parse_site_spec(spec)
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    if len(files) < 2:
        raise FlagshipError(f"{name}: в {directory} меньше двух .parquet")
    n_test = count if count is not None else max(1, int(len(files) * 0.15))
    if not 0 < n_test < len(files):
        raise FlagshipError(f"{name}: на проверку {n_test} из {len(files)} файлов")

    parts = []
    for path in files:
        frame = pd.read_parquet(path, columns=cols + ["Label"])
        frame["_file"] = os.path.basename(path)
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    X = frame[cols].replace([np.inf, -np.inf], np.nan).astype("float64")
    raw = Dataset(name, X, frame["Label"].astype(str).str.strip(), frame["_file"])

    test_files = set(os.path.basename(path) for path in files[-n_test:])
    is_test = raw.files.isin(test_files).to_numpy()
    # Лимиты — отдельно для обучения и проверки, чтобы проверка шла по своим файлам
    train_rows = np.flatnonzero(~is_test)
    test_rows = np.flatnonzero(is_test)
    train_rows = train_rows[_cap(raw.labels.iloc[train_rows].reset_index(drop=True),
                                 args.max_benign, args.max_attack, rng)]
    test_rows = test_rows[_cap(raw.labels.iloc[test_rows].reset_index(drop=True),
                               args.max_benign // 4, args.max_attack // 4, rng)]
    data = _subset(raw, np.concatenate([train_rows, test_rows]))
    train, calib = _split_calib(data, np.arange(len(train_rows)), rng)
    test = np.arange(len(train_rows), len(data.X))
    return Network(name, data, train, calib, test)


class Flagship:
    """IF (log1p + StandardScaler) на норме всех сетей и RF «атака / норма»."""

    def __init__(self, cols: list[str], args):
        self.cols = cols
        self.args = args

    def fit(self, networks: list[Network]) -> "Flagship":
        args = self.args
        benign_train = pd.concat(
            [n.data.X.iloc[n.train[~n.data.y[n.train]]] for n in networks], ignore_index=True
        )
        self.means = benign_train.mean()
        self.stds = benign_train.std()
        self.benign_rows = len(benign_train)

        X_benign = self.features(benign_train)
        self.scaler = make_scaler(log1p=True).fit(X_benign)
        self.iforest = IsolationForest(
            n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs
        ).fit(self.scaler.transform(X_benign))

        X_train = pd.concat([self.features(n.data.X.iloc[n.train]) for n in networks],
                            ignore_index=True)
        y_train = np.concatenate([n.data.y[n.train] for n in networks]).astype(int)
        params = {
            "n_estimators": args.rf_trees,
            "max_depth": args.rf_depth or None,
            "min_samples_leaf": args.rf_leaf,
            "max_samples": args.rf_max_samples,
            "class_weight": "balanced_subsample",
            "random_state": args.seed,
        }
        self.rf = RandomForestClassifier(**params, n_jobs=args.n_jobs).fit(X_train, y_train)
        # в bruteforce.json параметры хранятся с seed, как у train_bruteforce.py
        seed = params.pop("random_state")
        self.rf_params = {**params, "seed": seed}
        self.train_rows = len(X_train)

        # Пороги — по отложенной норме всех сетей, каждая сеть с равным весом
        empty = [n.name for n in networks if not len(n.calib)]
        if empty:
            raise FlagshipError("Нет нормы для порога у сетей: " + ", ".join(empty))
        calib = [n.data.X.iloc[n.calib] for n in networks]
        calib_if = [self.if_score(part) for part in calib]
        calib_rf = [self.rf_proba(part) for part in calib]
        self.calib_scores = np.concatenate(calib_if)
        self.calib_weights = np.concatenate([np.full(len(s), 1.0 / len(s)) for s in calib_if])
        self.threshold = _weighted_quantile(self.calib_scores, self.calib_weights, args.target_fpr)
        rf_quantile = _weighted_quantile(
            np.concatenate(calib_rf), self.calib_weights, 1 - args.rf_target_fpr
        )
        # Чуть выше квантиля (у RF много одинаковых вероятностей), но строго
        # меньше 1: load_bruteforce принимает только 0 < threshold < 1
        self.rf_threshold = float(min(np.nextafter(rf_quantile, np.inf), np.nextafter(1.0, 0.0)))
        self.calib_rows = len(self.calib_scores)
        return self

    def features(self, X: pd.DataFrame) -> pd.DataFrame:
        features, _ = prepare_features(X, self.cols, fill_values=self.means)
        return features

    def if_score(self, X: pd.DataFrame) -> np.ndarray:
        """decision_function: меньше — аномальнее (как в pipeline)."""
        return if_scores(self.iforest, self.scaler, None, self.features(X))

    def rf_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.rf.predict_proba(self.features(X))[:, 1]

    def evaluate(self, data: Dataset, rows: np.ndarray) -> dict:
        part = _subset(data, rows)
        return {
            "isolation_forest": _summary(evaluate_scores(
                part, -self.if_score(part.X), -self.threshold, self.args.target_fpr)),
            "supervised": _summary(evaluate_scores(
                part, self.rf_proba(part.X), float(np.nextafter(self.rf_threshold, -np.inf)),
                self.args.rf_target_fpr)),
        }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(values[order][min(np.searchsorted(cumulative, q), len(values) - 1)])


def _summary(metrics: dict) -> dict:
    return summarize_metrics(metrics, REPORT_KEYS)


def evaluate_model_dir(model_dir: str, data: Dataset, rows: np.ndarray, args) -> dict:
    """Готовая модель приложения (models/ или адаптированная) на тех же строках."""
    model, scaler, selector, cols = load_model_objects(model_dir)
    means, _ = load_normal_stats(cols, model_dir=model_dir)
    calibration = load_calibration(model_dir)
    part = _subset(data, rows)
    X, _ = prepare_features(part.X, cols, fill_values=means)
    scores = if_scores(model, scaler, selector, X)
    threshold = calibration.threshold if calibration is not None else 0.0
    result = {"isolation_forest": _summary(evaluate_scores(
        part, -scores, float(np.nextafter(-threshold, -np.inf)), args.target_fpr))}
    brute = load_bruteforce(model_dir)
    if brute is not None:
        X_brute, _ = prepare_features(part.X, list(brute.feature_cols), fill_values=means)
        result["supervised"] = _summary(evaluate_scores(
            part, brute.predict_proba(X_brute), float(np.nextafter(brute.threshold, -np.inf)),
            args.rf_target_fpr))
        result["supervised"]["display_name"] = brute.display_name
    return result


def save(model: Flagship, output_dir: str, networks: list[Network], report: dict, args) -> dict:
    if os.path.exists(output_dir) and args.force:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols = model.cols
    sources = [n.name for n in networks]

    joblib.dump(model.iforest, os.path.join(output_dir, "isolation_forest.pkl"), compress=3)
    joblib.dump(model.scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(list(cols), os.path.join(output_dir, "feature_cols_filtered.pkl"))
    stds = model.stds.where(np.isfinite(model.stds) & (model.stds > 0), 1.0)
    write_json(os.path.join(output_dir, STATS_FILE), {
        "source": ", ".join(sources),
        "rows": int(model.benign_rows),
        "created_at": created_at,
        "feature_cols": list(cols),
        "means": {c: float(v) for c, v in model.means.reindex(cols).fillna(0).items()},
        "stds": {c: float(v) for c, v in stds.reindex(cols).fillna(1).items()},
    })
    write_json(os.path.join(output_dir, CALIBRATION_FILE), {
        "threshold": model.threshold,
        "target_fpr": float(args.target_fpr),
        "rows": int(model.calib_rows),
        "quantiles": [
            _weighted_quantile(model.calib_scores, model.calib_weights, q)
            for q in np.linspace(0, 1, 101)
        ],
        "source": "flagship: benign holdout of " + ", ".join(sources),
        "created_at": created_at,
    })
    joblib.dump(model.rf, os.path.join(output_dir, "bruteforce.pkl"), compress=3)
    write_json(os.path.join(output_dir, "bruteforce.json"), {
        "model": "RandomForestClassifier",
        "created_at": created_at,
        "display_name": DISPLAY_NAME,
        "threshold": model.rf_threshold,
        "target_fpr": float(args.rf_target_fpr),
        "feature_cols": list(cols),
        "rows": int(model.train_rows),
        "params": model.rf_params,
        "adaptation": {"mode": "flagship", "networks": sources},
        "note": "Флагман: все атаки против нормы на " + ", ".join(sources) + ".",
    })
    metadata = {
        "version": "flagship",
        "created_at": created_at,
        "note": (
            "Одна модель на всех сетях (" + ", ".join(sources) + "), каждая с равным весом. "
            "Проверка «без одной сети» и на CIC-DDoS2019 — в lono и external."
        ),
        "networks": {
            n.name: {"train": int(len(n.train)), "calib": int(len(n.calib)),
                     "test": int(len(n.test)), "attacks_train": int(n.data.y[n.train].sum())}
            for n in networks
        },
        "log1p": True,
        "target_fpr": float(args.target_fpr),
        "threshold": model.threshold,
        "rf_target_fpr": float(args.rf_target_fpr),
        **report,
    }
    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    checksums = write_checksums(output_dir)
    return {"signed": "signature" in checksums, "metadata": metadata}


def _card_block(evaluation: dict) -> dict:
    """Формат карточки модели в приложении: {model: {adapted: метрики}}."""
    block = {"isolation_forest": {"adapted": evaluation["isolation_forest"]}}
    block["supervised"] = {"display_name": DISPLAY_NAME, "adapted": evaluation["supervised"]}
    return block


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def format_table(title: str, rows: dict[str, dict]) -> str:
    lines = [title, f"{'сеть':<30}{'IF AUC':>8}{'IF полн.':>10}{'IF FPR':>9}"
             f"{'RF AUC':>8}{'RF полн.':>10}{'RF FPR':>9}"]
    for name, result in rows.items():
        iforest, rf = result["isolation_forest"], result.get("supervised") or {}
        lines.append(
            f"{name:<30}{(iforest['auc'] or 0):>8.3f}{_pct(iforest['recall']):>10}"
            f"{_pct(iforest['fpr']):>9}{(rf.get('auc') or 0):>8.3f}"
            f"{_pct(rf.get('recall')):>10}{_pct(rf.get('fpr')):>9}"
        )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = resolve_model_dir(args.output_dir)
    if os.path.exists(output_dir) and os.listdir(output_dir) and not args.force:
        print(f"Ошибка: {output_dir} уже существует. Добавьте --force.")
        return 2
    rng = np.random.default_rng(args.seed)
    _, _, _, cols = load_model_objects(resolve_model_dir("models"))
    cols = list(cols)

    try:
        paths = resolve_datasets(None)
        missing = [name for name in TRAIN_NETWORKS if name not in paths]
        if missing:
            raise FlagshipError("Нет в кэше kagglehub: " + ", ".join(missing))
        networks = []
        for name in TRAIN_NETWORKS:
            print(f"Загрузка {name}…", flush=True)
            networks.append(load_cic(name, paths[name], cols, args, rng))
        for spec in args.site:
            print(f"Загрузка {spec.split('=')[0]}…", flush=True)
            networks.append(load_site(spec, cols, args, rng))
        compare = dict(spec.split("=", 1) for spec in args.compare)
    except (FlagshipError, ValueError) as exc:
        print(f"Ошибка: {exc}")
        return 2

    for n in networks:
        print(f"  {n.name}: обучение {len(n.train):,} (атак {int(n.data.y[n.train].sum()):,}), "
              f"порог {len(n.calib):,}, проверка {len(n.test):,} "
              f"(атак {int(n.data.y[n.test].sum()):,})")

    report: dict = {"lono": {}, "in_domain": {}, "baseline": {}, "adapted": {}, "external": {}}
    if not args.no_lono:
        for held in networks:
            print(f"Без сети {held.name}: обучение на остальных…", flush=True)
            others = [n for n in networks if n is not held]
            report["lono"][held.name] = Flagship(cols, args).fit(others).evaluate(held.data, held.test)

    print("Итоговая модель на всех сетях…", flush=True)
    model = Flagship(cols, args).fit(networks)
    base_dir = resolve_model_dir("models")
    for n in networks:
        report["in_domain"][n.name] = model.evaluate(n.data, n.test)
        report["baseline"][n.name] = evaluate_model_dir(base_dir, n.data, n.test, args)
        if n.name in compare:
            report["adapted"][n.name] = evaluate_model_dir(compare[n.name], n.data, n.test, args)

    if EXTERNAL_NETWORK in paths:
        print(f"Проверка на {EXTERNAL_NETWORK}…", flush=True)
        raw = load_dataset(EXTERNAL_NETWORK, paths[EXTERNAL_NETWORK], cols,
                           args.max_external, args.max_external, args.seed)
        external = _subset(raw, _cap(raw.labels, args.max_external, args.max_external, rng))
        rows = np.arange(len(external.X))
        report["external"][EXTERNAL_NETWORK] = model.evaluate(external, rows)
        report["external"][f"{EXTERNAL_NETWORK} (базовая)"] = evaluate_model_dir(
            base_dir, external, rows, args)

    test_union = Dataset(
        "all", pd.concat([n.data.X.iloc[n.test] for n in networks], ignore_index=True),
        pd.concat([n.data.labels.iloc[n.test] for n in networks], ignore_index=True),
        pd.concat([n.data.files.iloc[n.test] for n in networks], ignore_index=True),
    )
    combined = model.evaluate(test_union, np.arange(len(test_union.X)))
    report["evaluation_test"] = {"files": [f"{n.name}: отложенная часть" for n in networks],
                                 **_card_block(combined)}

    result = save(model, output_dir, networks, report, args)
    if report["lono"]:
        print("\n" + format_table("Без одной сети (модель её не видела):", report["lono"]))
    print("\n" + format_table("Флагман на отложенной части каждой сети:", report["in_domain"]))
    print("\n" + format_table("Базовая модель (models/) на тех же строках:", report["baseline"]))
    if report["adapted"]:
        print("\n" + format_table("Адаптированные под сеть модели:", report["adapted"]))
    if report["external"]:
        print("\n" + format_table("Чужая сеть:", report["external"]))
    print(f"\nСохранено в {output_dir}" + (" (подписано)" if result["signed"] else " (НЕ подписано)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
