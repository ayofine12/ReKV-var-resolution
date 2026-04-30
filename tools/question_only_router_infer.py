#!/usr/bin/env python3
"""
Inference helper for a saved question-only router bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd


LABEL_112 = "route_112"
LABEL_224 = "route_224"
LABEL_TIE = "tie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a saved question-only router on one or more questions.")
    parser.add_argument("--router-bundle", type=Path, required=True, help="Path to a saved router bundle.")
    parser.add_argument("--question", action="append", default=[], help="Question text. Can be repeated.")
    parser.add_argument(
        "--question-file",
        type=Path,
        default=None,
        help="Optional text/csv/json/jsonl file containing questions.",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional path to save predictions as CSV.",
    )
    return parser.parse_args()


def load_questions(question_args: List[str], question_file: Path | None) -> List[str]:
    questions = list(question_args)
    if question_file is None:
        if not questions:
            raise ValueError("Provide at least one --question or a --question-file.")
        return questions

    suffix = question_file.suffix.lower()
    if suffix in {".txt", ".md"}:
        with question_file.open() as fh:
            questions.extend([line.strip() for line in fh if line.strip()])
    elif suffix == ".csv":
        with question_file.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if "question" not in reader.fieldnames:
                raise ValueError("CSV question file must contain a 'question' column.")
            questions.extend([row["question"].strip() for row in reader if row["question"].strip()])
    elif suffix == ".json":
        with question_file.open() as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    questions.append(item.strip())
                elif isinstance(item, dict) and "question" in item:
                    questions.append(str(item["question"]).strip())
                else:
                    raise ValueError("JSON question list items must be strings or dicts with 'question'.")
        else:
            raise ValueError("JSON question file must be a list.")
    elif suffix == ".jsonl":
        with question_file.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    questions.append(item.strip())
                elif isinstance(item, dict) and "question" in item:
                    questions.append(str(item["question"]).strip())
                else:
                    raise ValueError("JSONL items must be strings or dicts with 'question'.")
    else:
        raise ValueError(f"Unsupported question file format: {question_file}")

    questions = [q for q in questions if q]
    if not questions:
        raise ValueError("No valid questions found.")
    return questions


def route_question(bundle: Dict[str, object], question: str) -> Dict[str, object]:
    pipeline = bundle["pipeline"]
    threshold = float(bundle["min_confidence"])
    default_fs = str(bundle["default_fs"])
    proba = pipeline.predict_proba(pd.DataFrame({"question": [question]}))[0]
    classes = list(pipeline.named_steps["model"].classes_)
    score_map = {cls: float(prob) for cls, prob in zip(classes, proba)}
    p112 = score_map.get(LABEL_112, 0.0)
    p224 = score_map.get(LABEL_224, 0.0)
    if p112 >= p224:
        pred_label = LABEL_112
        pred_conf = p112
    else:
        pred_label = LABEL_224
        pred_conf = p224
    if pred_conf < threshold:
        pred_label = LABEL_TIE
    routed_fs = default_fs if pred_label == LABEL_TIE else ("112" if pred_label == LABEL_112 else "224")
    return {
        "question": question,
        "p_route_112": p112,
        "p_route_224": p224,
        "pred_label": pred_label,
        "pred_confidence": pred_conf,
        "default_fs": default_fs,
        "threshold": threshold,
        "routed_fs": routed_fs,
    }


def save_predictions(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[saved_predictions] {path}")


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.router_bundle)
    questions = load_questions(args.question, args.question_file)
    rows = [route_question(bundle, question) for question in questions]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    if args.save_predictions is not None:
        save_predictions(args.save_predictions, rows)


if __name__ == "__main__":
    main()
