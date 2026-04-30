#!/usr/bin/env python3
"""
Evaluate a saved question-only router bundle on a holdout dataset with fs112/fs224 results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import pandas as pd


LABEL_112 = "route_112"
LABEL_224 = "route_224"
LABEL_TIE = "tie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved router bundle on a holdout 112-vs-224 dataset.")
    parser.add_argument("--router-bundle", type=Path, required=True, help="Path to a saved router bundle.")
    parser.add_argument("--csv-112", nargs="+", required=True, help="Holdout fs112 result CSV(s).")
    parser.add_argument("--csv-224", nargs="+", required=True, help="Holdout fs224 result CSV(s).")
    parser.add_argument(
        "--override-default-fs",
        choices=["112", "224"],
        default=None,
        help="Optional override for the bundle default fs on the holdout dataset.",
    )
    parser.add_argument(
        "--override-threshold",
        type=float,
        default=None,
        help="Optional override for the bundle confidence threshold on the holdout dataset.",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional path to save per-question routed predictions as CSV.",
    )
    return parser.parse_args()


def result_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        row["video_id"],
        row["question"],
        row.get("choices", ""),
        row.get("correct_choice", ""),
    )


def normalize_acc(raw: str) -> float:
    value = float(raw)
    if value > 1.0:
        return value / 100.0
    return value


def load_side(paths: Sequence[str]) -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    rows: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for path_str in paths:
        path = Path(path_str)
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows[result_key(row)] = dict(row)
    return rows


def save_predictions(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[saved_predictions] {path}")


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.router_bundle)
    pipeline = bundle["pipeline"]
    threshold = float(args.override_threshold if args.override_threshold is not None else bundle["min_confidence"])
    default_fs = str(args.override_default_fs if args.override_default_fs is not None else bundle["default_fs"])

    rows112 = load_side(args.csv_112)
    rows224 = load_side(args.csv_224)
    keys = sorted(set(rows112) & set(rows224))
    if not keys:
        raise ValueError("No overlapping questions between the provided 112 and 224 CSVs.")

    questions = [key[1] for key in keys]
    proba = pipeline.predict_proba(pd.DataFrame({"question": questions}))
    classes = list(pipeline.named_steps["model"].classes_)
    idx112 = classes.index(LABEL_112) if LABEL_112 in classes else None
    idx224 = classes.index(LABEL_224) if LABEL_224 in classes else None

    pred_rows: List[Dict[str, object]] = []
    routed_accs: List[float] = []
    base_acc_112: List[float] = []
    base_acc_224: List[float] = []
    oracle_accs: List[float] = []
    pred_counter: Counter[str] = Counter()
    gold_counter: Counter[str] = Counter()

    for i, key in enumerate(keys):
        row112 = rows112[key]
        row224 = rows224[key]
        acc112 = normalize_acc(row112["qa_acc"])
        acc224 = normalize_acc(row224["qa_acc"])
        base_acc_112.append(acc112)
        base_acc_224.append(acc224)
        oracle_accs.append(max(acc112, acc224))
        if acc112 > acc224:
            gold_label = LABEL_112
        elif acc224 > acc112:
            gold_label = LABEL_224
        else:
            gold_label = LABEL_TIE
        gold_counter[gold_label] += 1

        p112 = float(proba[i, idx112]) if idx112 is not None else 0.0
        p224 = float(proba[i, idx224]) if idx224 is not None else 0.0
        if p112 >= p224:
            pred_label = LABEL_112
            pred_conf = p112
        else:
            pred_label = LABEL_224
            pred_conf = p224
        if pred_conf < threshold:
            pred_label = LABEL_TIE
        pred_counter[pred_label] += 1

        if pred_label == LABEL_112:
            routed_fs = "112"
            routed_acc = acc112
        elif pred_label == LABEL_224:
            routed_fs = "224"
            routed_acc = acc224
        else:
            routed_fs = default_fs
            routed_acc = acc112 if default_fs == "112" else acc224
        routed_accs.append(routed_acc)

        pred_rows.append(
            {
                "video_id": row112["video_id"],
                "question": row112["question"],
                "task": row112.get("task", ""),
                "acc112": acc112,
                "acc224": acc224,
                "gold_label": gold_label,
                "p_route_112": p112,
                "p_route_224": p224,
                "pred_label": pred_label,
                "pred_confidence": pred_conf,
                "default_fs": default_fs,
                "threshold": threshold,
                "routed_fs": routed_fs,
                "routed_acc": routed_acc,
            }
        )

    default_acc = sum(base_acc_112) / len(base_acc_112) if default_fs == "112" else sum(base_acc_224) / len(base_acc_224)
    routed_acc = sum(routed_accs) / len(routed_accs)
    switch_ratio = sum(1 for row in pred_rows if row["pred_label"] != LABEL_TIE) / len(pred_rows)

    print("[summary]")
    print(f"n_examples: {len(keys)}")
    print(f"default_fs: {default_fs}")
    print(f"threshold: {threshold:.2f}")
    print(f"base_acc_112: {100 * (sum(base_acc_112) / len(base_acc_112)):.2f}")
    print(f"base_acc_224: {100 * (sum(base_acc_224) / len(base_acc_224)):.2f}")
    print(f"default_acc: {100 * default_acc:.2f}")
    print(f"question_oracle_acc: {100 * (sum(oracle_accs) / len(oracle_accs)):.2f}")
    print(f"routed_acc: {100 * routed_acc:.2f}")
    print(f"gain_vs_default: {100 * (routed_acc - default_acc):.2f}p")
    print(f"gain_vs_112: {100 * (routed_acc - (sum(base_acc_112) / len(base_acc_112))):.2f}p")
    print(f"gain_vs_224: {100 * (routed_acc - (sum(base_acc_224) / len(base_acc_224))):.2f}p")
    print(f"switch_ratio: {100 * switch_ratio:.2f}")
    print(f"gold_distribution: {json.dumps(gold_counter, ensure_ascii=False)}")
    print(f"prediction_distribution: {json.dumps(pred_counter, ensure_ascii=False)}")

    if args.save_predictions is not None:
        save_predictions(args.save_predictions, pred_rows)


if __name__ == "__main__":
    main()
