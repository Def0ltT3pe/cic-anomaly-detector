"""CLI: пакетный поиск аномалий в CSV/Parquet и текстовый отчёт."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.bruteforce import BruteForceError, load_bruteforce
from modules.calibration import CalibrationError, load_calibration
from modules.explainer import explain_anomaly
from modules.model_loader import ModelLoadError, load_model_objects, load_normal_stats
from modules.pipeline import FeatureValidationError, detect_anomalies


def load_data(path: str) -> pd.DataFrame:
    """Загружает CSV или Parquet по расширению."""
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Поддерживаются только файлы .csv и .parquet.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Поиск аномалий сетевого трафика")
    parser.add_argument("--input", required=True, help="CSV или Parquet с признаками")
    parser.add_argument("--output", default="anomaly_report.txt", help="Путь к отчёту")
    parser.add_argument("--model-dir", default=None, help="Каталог с .pkl (по умолчанию models/)")
    parser.add_argument("--use_llm", action="store_true", help="Добавить пояснения Gemini")
    parser.add_argument(
        "--max_anomalies",
        type=int,
        default=20,
        help="Сколько аномалий описать в отчёте (>= 1)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.max_anomalies < 1:
        print("Ошибка: --max_anomalies должен быть >= 1")
        return 2
    if not os.path.isfile(args.input):
        print(f"Ошибка: файл не найден: {args.input}")
        return 2

    try:
        model, scaler, selector, feature_cols = load_model_objects(args.model_dir)
        means, stds = load_normal_stats(feature_cols, model_dir=args.model_dir)
    except ModelLoadError as exc:
        print(f"Ошибка загрузки модели: {exc}")
        return 2

    try:
        calibration = load_calibration(args.model_dir)
    except CalibrationError as exc:
        print(f"ВНИМАНИЕ: {exc} — используется встроенный порог модели.")
        calibration = None

    try:
        brute = load_bruteforce(args.model_dir)
    except BruteForceError as exc:
        print(f"ВНИМАНИЕ: BruteForce-модель не загружена: {exc}")
        brute = None

    try:
        df = load_data(args.input)
    except Exception as exc:
        print(f"Ошибка чтения {args.input}: {exc}")
        return 2

    try:
        result = detect_anomalies(
            df,
            model,
            scaler,
            selector,
            feature_cols,
            fill_values=means,
            calibration=calibration,
        )
    except FeatureValidationError as exc:
        print(f"Ошибка данных: {exc}")
        return 2

    print(f"Загружено записей: {result.total}")
    print(f"Найдено аномалий: {result.n_anomalies} ({result.share:.2%})")

    brute_probs = None
    brute_mask = None
    if brute is not None:
        try:
            brute_probs = brute.predict_proba(result.X)
            brute_mask = brute_probs >= brute.threshold
            print(
                f"{brute.display_name}-подозрений: {int(brute_mask.sum())} "
                f"(порог {brute.threshold:.3f})"
            )
        except BruteForceError as exc:
            print(f"ВНИМАНИЕ: BruteForce-модель не применилась: {exc}")

    explainer = None
    if args.use_llm:
        try:
            from modules.llm_explain import enhance_report_with_gemini, is_llm_configured

            if is_llm_configured():
                explainer = enhance_report_with_gemini
            else:
                print("ВНИМАНИЕ: GEMINI_API_KEY не задан — отчёт будет без пояснений ИИ.")
        except Exception as exc:
            print(f"ВНИМАНИЕ: LLM недоступен ({exc}) — отчёт будет без пояснений ИИ.")

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    selected = result.top_anomalies(args.max_anomalies)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("ОТЧЁТ ОБ АНОМАЛИЯХ\n")
        handle.write(f"Файл: {os.path.abspath(args.input)}\n")
        handle.write(f"Всего записей: {result.total}\n")
        handle.write(f"Найдено аномалий: {result.n_anomalies} ({result.share:.2%})\n")
        if result.threshold is not None:
            handle.write(f"Порог score: <= {result.threshold:.4f}\n")
        if brute_mask is not None:
            handle.write(
                f"{brute.display_name}-подозрений: {int(brute_mask.sum())} "
                f"(порог {brute.threshold:.3f})\n"
            )
        if result.info.get("inf_replaced") or result.info.get("nan_filled"):
            handle.write(
                f"Заменено inf: {result.info.get('inf_replaced', 0)}, "
                f"пропусков: {result.info.get('nan_filled', 0)}\n"
            )
        handle.write("\n")

        for index in selected:
            index = int(index)
            stats_text = explain_anomaly(result.X.iloc[index], means, stds)
            label = (
                str(result.df.iloc[index]["Label"])
                if "Label" in result.df.columns
                else "неизвестно"
            )
            handle.write(
                f"Аномалия # {index} (истинная метка: {label}, "
                f"score: {result.scores[index]:.4f})\n"
            )
            if result.percentiles is not None:
                handle.write(
                    f"Аномальнее {result.percentiles[index]:.1f}% нормального трафика\n"
                )
            handle.write("Статистические отклонения:\n")
            handle.write(stats_text + "\n")

            if explainer is not None:
                try:
                    handle.write("Пояснение ИИ:\n")
                    handle.write(explainer(stats_text) + "\n")
                except Exception as exc:
                    handle.write(f"Пояснение ИИ: ошибка — {exc}\n")

            handle.write("=" * 80 + "\n\n")

        if brute_mask is not None and brute_mask.any():
            handle.write(f"ПОДОЗРЕНИЯ: {brute.display_name.upper()} (supervised)\n")
            flagged = np.flatnonzero(brute_mask)
            brute_top = flagged[np.argsort(-brute_probs[flagged])][: args.max_anomalies]
            for index in brute_top:
                index = int(index)
                label = (
                    str(result.df.iloc[index]["Label"])
                    if "Label" in result.df.columns
                    else "неизвестно"
                )
                handle.write(
                    f"Строка # {index} (p={brute_probs[index]:.4f}, метка: {label})\n"
                )
                handle.write("Статистические отклонения:\n")
                handle.write(explain_anomaly(result.X.iloc[index], means, stds) + "\n")
                handle.write("=" * 80 + "\n\n")

    print(f"Отчёт сохранён: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
