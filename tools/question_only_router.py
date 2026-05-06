#!/usr/bin/env python3
"""
Feasibility test for a text-only fs112 vs fs224 router.

This script trains a text classifier on either question strings only or
question+choices text, then evaluates whether routing between two result sets
can improve end-to-end QA accuracy.
It supports one or more CSVs per side, which is convenient for datasets such
as MLVU that are split into front/back subsets.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline


LABEL_112 = "route_112"
LABEL_224 = "route_224"
LABEL_TIE = "tie"


@dataclass(frozen=True)
class RouterExample:
    video_id: str
    question: str
    task: str
    choices: str
    correct_choice: str
    acc112: float
    acc224: float
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a text-only 112-vs-224 router.")
    parser.add_argument("--csv-112", nargs="+", required=True, help="One or more result CSVs for fs112.")
    parser.add_argument("--csv-224", nargs="+", required=True, help="One or more result CSVs for fs224.")
    parser.add_argument(
        "--input-mode",
        choices=["question", "question_choices"],
        default="question",
        help="Router text input. 'question_choices' uses question plus multiple-choice options.",
    )
    parser.add_argument(
        "--routing-mode",
        choices=["binary_abstain", "multiclass"],
        default="binary_abstain",
        help="Training mode. 'binary_abstain' trains on 112-vs-224 only and falls back to tie/default on low confidence.",
    )
    parser.add_argument(
        "--default-fs",
        choices=["112", "224", "auto"],
        default="auto",
        help="Fallback fs used for tie predictions. 'auto' picks the higher-accuracy side.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of grouped CV folds.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="If max class probability is below this threshold, abstain to tie/default.",
    )
    parser.add_argument(
        "--max-word-features",
        type=int,
        default=20000,
        help="Maximum number of word n-gram TF-IDF features.",
    )
    parser.add_argument(
        "--max-char-features",
        type=int,
        default=30000,
        help="Maximum number of char n-gram TF-IDF features.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path to save a raw sklearn pipeline trained on the full dataset.",
    )
    parser.add_argument(
        "--save-router-bundle",
        type=Path,
        default=None,
        help="Optional path to save a deployable router bundle with model + threshold + metadata.",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional path to save per-question CV predictions as CSV.",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=20,
        help="Number of top text features to print for each routed class when saving a full model.",
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


def load_result_side(paths: Sequence[str], side_name: str) -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    merged: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for path_str in paths:
        path = Path(path_str)
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = result_key(row)
                if key in merged:
                    raise ValueError(f"Duplicate key for {side_name}: {key} from {path}")
                row = dict(row)
                row["_source_path"] = str(path)
                merged[key] = row
    return merged


def build_examples(paths112: Sequence[str], paths224: Sequence[str]) -> List[RouterExample]:
    rows112 = load_result_side(paths112, "112")
    rows224 = load_result_side(paths224, "224")
    common_keys = sorted(set(rows112) & set(rows224))
    if not common_keys:
        raise ValueError("No overlapping questions between the provided 112 and 224 CSVs.")

    examples: List[RouterExample] = []
    for key in common_keys:
        r112 = rows112[key]
        r224 = rows224[key]
        acc112 = normalize_acc(r112["qa_acc"])
        acc224 = normalize_acc(r224["qa_acc"])
        if acc112 > acc224:
            label = LABEL_112
        elif acc224 > acc112:
            label = LABEL_224
        else:
            label = LABEL_TIE
        examples.append(
            RouterExample(
                video_id=r112["video_id"],
                question=r112["question"],
                task=r112.get("task", ""),
                choices=r112.get("choices", ""),
                correct_choice=r112.get("correct_choice", ""),
                acc112=acc112,
                acc224=acc224,
                label=label,
            )
        )
    return examples


def choose_default_fs(examples: Sequence[RouterExample], default_fs: str) -> str:
    if default_fs in {"112", "224"}:
        return default_fs
    mean112 = float(np.mean([ex.acc112 for ex in examples]))
    mean224 = float(np.mean([ex.acc224 for ex in examples]))
    return "112" if mean112 >= mean224 else "224"


def choose_label(acc112: float, acc224: float) -> str:
    if acc112 > acc224:
        return LABEL_112
    if acc224 > acc112:
        return LABEL_224
    return LABEL_TIE


def parse_choices(raw: str) -> List[str]:
    if not raw:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
    return [raw]


def format_choices(raw: str) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines: List[str] = []
    for idx, choice in enumerate(parse_choices(raw)):
        prefix = labels[idx] if idx < len(labels) else str(idx + 1)
        lines.append(f"({prefix}) {choice}")
    return "\n".join(lines)


def build_router_text(question: str, choices: str, input_mode: str) -> str:
    if input_mode == "question":
        return question
    if input_mode == "question_choices":
        formatted_choices = format_choices(choices)
        if formatted_choices:
            return f"Question:\n{question}\n\nChoices:\n{formatted_choices}"
        return f"Question:\n{question}"
    raise ValueError(f"Unsupported input mode: {input_mode}")


def build_pipeline(max_word_features: int, max_char_features: int) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    max_features=max_word_features,
                    sublinear_tf=True,
                ),
                "router_text",
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    max_features=max_char_features,
                    sublinear_tf=True,
                ),
                "router_text",
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("features", features), ("model", model)])


def build_training_frame(examples: Sequence[RouterExample], input_mode: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": [ex.video_id for ex in examples],
            "question": [ex.question for ex in examples],
            "router_text": [build_router_text(ex.question, ex.choices, input_mode) for ex in examples],
            "task": [ex.task for ex in examples],
            "choices": [ex.choices for ex in examples],
            "correct_choice": [ex.correct_choice for ex in examples],
            "acc112": [ex.acc112 for ex in examples],
            "acc224": [ex.acc224 for ex in examples],
            "label": [ex.label for ex in examples],
        }
    )


def select_fit_frame(df: pd.DataFrame, routing_mode: str) -> pd.DataFrame:
    if routing_mode == "binary_abstain":
        fit_df = df[df["label"] != LABEL_TIE]
    else:
        fit_df = df
    if fit_df["label"].nunique() < 2:
        fit_df = df
    if fit_df["label"].nunique() < 2:
        raise ValueError("Not enough label diversity to train a classifier.")
    return fit_df


def select_splitter(labels: Sequence[str], groups: Sequence[str], n_splits: int):
    group_count = len(set(groups))
    if group_count < n_splits:
        n_splits = group_count
    if n_splits < 2:
        raise ValueError("Need at least two unique video_ids for grouped cross-validation.")
    label_counts = Counter(labels)
    if min(label_counts.values()) >= n_splits:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits
    return GroupKFold(n_splits=n_splits), n_splits


def route_acc_for_prediction(pred_label: str, ex: RouterExample, default_fs: str) -> float:
    if pred_label == LABEL_112:
        return ex.acc112
    if pred_label == LABEL_224:
        return ex.acc224
    return ex.acc112 if default_fs == "112" else ex.acc224


def question_oracle_acc(examples: Sequence[RouterExample]) -> float:
    return float(np.mean([max(ex.acc112, ex.acc224) for ex in examples]))


def task_oracle_acc(examples: Sequence[RouterExample], default_fs: str) -> float:
    grouped: Dict[str, List[RouterExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.task, []).append(ex)
    best_for_task: Dict[str, str] = {}
    for task, task_examples in grouped.items():
        avg112 = float(np.mean([ex.acc112 for ex in task_examples]))
        avg224 = float(np.mean([ex.acc224 for ex in task_examples]))
        if avg112 > avg224:
            best_for_task[task] = LABEL_112
        elif avg224 > avg112:
            best_for_task[task] = LABEL_224
        else:
            best_for_task[task] = LABEL_TIE
    return float(np.mean([route_acc_for_prediction(best_for_task[ex.task], ex, default_fs) for ex in examples]))


def evaluate_cv(
    examples: Sequence[RouterExample],
    pipeline: Pipeline,
    input_mode: str,
    default_fs: str,
    min_confidence: float,
    n_splits: int,
    routing_mode: str,
):
    df = build_training_frame(examples, input_mode)

    splitter, n_splits = select_splitter(df["label"].tolist(), df["video_id"].tolist(), n_splits)
    all_true: List[str] = []
    all_pred: List[str] = []
    all_probs: List[float] = []
    routed_accs: List[float] = []
    prediction_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df, df["label"], groups=df["video_id"]), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        fit_df = select_fit_frame(train_df, routing_mode)
        pipeline.fit(fit_df[["router_text"]], fit_df["label"])
        proba = pipeline.predict_proba(test_df[["router_text"]])
        classes = list(pipeline.named_steps["model"].classes_)
        max_indices = np.argmax(proba, axis=1)
        raw_preds = [classes[idx] for idx in max_indices]
        max_probs = proba[np.arange(len(test_df)), max_indices]

        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            pred = raw_preds[row_idx]
            if max_probs[row_idx] < min_confidence:
                pred = LABEL_TIE
            ex = examples[row.name]
            routed_acc = route_acc_for_prediction(pred, ex, default_fs)
            all_true.append(row["label"])
            all_pred.append(pred)
            all_probs.append(float(max_probs[row_idx]))
            routed_accs.append(routed_acc)
            prediction_rows.append(
                {
                    "fold": fold_idx,
                    "input_mode": input_mode,
                    "video_id": row["video_id"],
                    "question": row["question"],
                    "choices": row["choices"],
                    "task": row["task"],
                    "gold_label": row["label"],
                    "pred_label": pred,
                    "pred_confidence": float(max_probs[row_idx]),
                    "acc112": row["acc112"],
                    "acc224": row["acc224"],
                    "routed_acc": routed_acc,
                }
            )

    base112 = float(np.mean(df["acc112"]))
    base224 = float(np.mean(df["acc224"]))
    default_acc = base112 if default_fs == "112" else base224
    routed_acc = float(np.mean(routed_accs))
    oracle = question_oracle_acc(examples)
    task_oracle = task_oracle_acc(examples, default_fs)
    switched_mask = [pred != LABEL_TIE for pred in all_pred]
    switched_ratio = float(np.mean(switched_mask))

    return {
        "n_examples": len(examples),
        "n_unique_videos": len(set(df["video_id"])),
        "n_splits": n_splits,
        "input_mode": input_mode,
        "routing_mode": routing_mode,
        "default_fs": default_fs,
        "base_acc_112": base112,
        "base_acc_224": base224,
        "default_acc": default_acc,
        "question_oracle_acc": oracle,
        "task_oracle_acc": task_oracle,
        "cv_routed_acc": routed_acc,
        "cv_gain_vs_default": routed_acc - default_acc,
        "cv_gain_vs_112": routed_acc - base112,
        "cv_gain_vs_224": routed_acc - base224,
        "switch_ratio": switched_ratio,
        "label_distribution": Counter(df["label"]),
        "prediction_distribution": Counter(all_pred),
        "classification_report": classification_report(
            all_true,
            all_pred,
            digits=4,
            zero_division=0,
        ),
        "prediction_rows": prediction_rows,
    }


def print_summary(report: Dict[str, object]) -> None:
    print("[summary]")
    print(f"n_examples: {report['n_examples']}")
    print(f"n_unique_videos: {report['n_unique_videos']}")
    print(f"n_splits: {report['n_splits']}")
    print(f"input_mode: {report['input_mode']}")
    print(f"routing_mode: {report['routing_mode']}")
    print(f"default_fs: {report['default_fs']}")
    print(f"base_acc_112: {100 * report['base_acc_112']:.2f}")
    print(f"base_acc_224: {100 * report['base_acc_224']:.2f}")
    print(f"default_acc: {100 * report['default_acc']:.2f}")
    print(f"question_oracle_acc: {100 * report['question_oracle_acc']:.2f}")
    print(f"task_oracle_acc: {100 * report['task_oracle_acc']:.2f}")
    print(f"cv_routed_acc: {100 * report['cv_routed_acc']:.2f}")
    print(f"cv_gain_vs_default: {100 * report['cv_gain_vs_default']:.2f}p")
    print(f"cv_gain_vs_112: {100 * report['cv_gain_vs_112']:.2f}p")
    print(f"cv_gain_vs_224: {100 * report['cv_gain_vs_224']:.2f}p")
    print(f"switch_ratio: {100 * report['switch_ratio']:.2f}")
    print(f"label_distribution: {json.dumps(report['label_distribution'], ensure_ascii=False)}")
    print(f"prediction_distribution: {json.dumps(report['prediction_distribution'], ensure_ascii=False)}")
    print("\n[classification_report]")
    print(report["classification_report"])


def save_predictions(path: Path, prediction_rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(path, index=False)
    print(f"[saved_predictions] {path}")


def print_top_features(pipeline: Pipeline, top_k: int) -> None:
    model: LogisticRegression = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["features"].get_feature_names_out()
    classes = list(model.classes_)
    for class_idx, class_name in enumerate(classes):
        coef = model.coef_[class_idx]
        top_indices = np.argsort(coef)[-top_k:][::-1]
        print(f"\n[top_features] {class_name}")
        for idx in top_indices:
            print(f"{feature_names[idx]}: {coef[idx]:.4f}")


def train_full_model(
    examples: Sequence[RouterExample],
    pipeline: Pipeline,
    input_mode: str,
    routing_mode: str,
) -> Pipeline:
    df = build_training_frame(examples, input_mode)
    fit_df = select_fit_frame(df, routing_mode)
    pipeline.fit(fit_df[["router_text"]], fit_df["label"])
    return pipeline


def save_raw_model(pipeline: Pipeline, save_model: Path, top_k_features: int) -> None:
    save_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_model)
    print(f"[saved_model] {save_model}")
    print_top_features(pipeline, top_k_features)


def save_router_bundle(
    bundle_path: Path,
    pipeline: Pipeline,
    examples: Sequence[RouterExample],
    input_mode: str,
    routing_mode: str,
    default_fs: str,
    min_confidence: float,
    csv_112: Sequence[str],
    csv_224: Sequence[str],
) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "input_mode": input_mode,
        "feature_column": "router_text",
        "routing_mode": routing_mode,
        "default_fs": default_fs,
        "min_confidence": float(min_confidence),
        "labels": [LABEL_112, LABEL_224, LABEL_TIE],
        "source_csv_112": list(csv_112),
        "source_csv_224": list(csv_224),
        "n_examples": len(examples),
        "n_unique_videos": len({ex.video_id for ex in examples}),
        "base_acc_112": float(np.mean([ex.acc112 for ex in examples])),
        "base_acc_224": float(np.mean([ex.acc224 for ex in examples])),
        "label_distribution": dict(Counter(ex.label for ex in examples)),
    }
    joblib.dump(bundle, bundle_path)
    print(f"[saved_router_bundle] {bundle_path}")
    print(f"[bundle_default_fs] {default_fs}")
    print(f"[bundle_threshold] {min_confidence:.2f}")


def main() -> None:
    args = parse_args()
    examples = build_examples(args.csv_112, args.csv_224)
    default_fs = choose_default_fs(examples, args.default_fs)
    pipeline = build_pipeline(args.max_word_features, args.max_char_features)
    report = evaluate_cv(
        examples=examples,
        pipeline=pipeline,
        input_mode=args.input_mode,
        default_fs=default_fs,
        min_confidence=args.min_confidence,
        n_splits=args.n_splits,
        routing_mode=args.routing_mode,
    )
    print_summary(report)

    if args.save_predictions is not None:
        save_predictions(args.save_predictions, report["prediction_rows"])
    if args.save_model is not None or args.save_router_bundle is not None:
        full_pipeline = build_pipeline(args.max_word_features, args.max_char_features)
        full_pipeline = train_full_model(
            examples=examples,
            pipeline=full_pipeline,
            input_mode=args.input_mode,
            routing_mode=args.routing_mode,
        )
        if args.save_model is not None:
            save_raw_model(full_pipeline, args.save_model, args.top_k_features)
        if args.save_router_bundle is not None:
            save_router_bundle(
                bundle_path=args.save_router_bundle,
                pipeline=full_pipeline,
                examples=examples,
                input_mode=args.input_mode,
                routing_mode=args.routing_mode,
                default_fs=default_fs,
                min_confidence=args.min_confidence,
                csv_112=args.csv_112,
                csv_224=args.csv_224,
            )


if __name__ == "__main__":
    main()
