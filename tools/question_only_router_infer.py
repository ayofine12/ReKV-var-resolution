#!/usr/bin/env python3
"""
Inference helper for a saved text-only router bundle.
"""

from __future__ import annotations

import argparse
import ast
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
    parser = argparse.ArgumentParser(description="Run a saved text-only router on one or more questions.")
    parser.add_argument("--router-bundle", type=Path, required=True, help="Path to a saved router bundle.")
    parser.add_argument("--question", action="append", default=[], help="Question text. Can be repeated.")
    parser.add_argument(
        "--choices",
        action="append",
        default=[],
        help="Choices for a --question. Can be repeated; aligns by position with --question.",
    )
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


def load_records(question_args: List[str], choices_args: List[str], question_file: Path | None) -> List[Dict[str, str]]:
    records = [
        {"question": question, "choices": choices_args[idx] if idx < len(choices_args) else ""}
        for idx, question in enumerate(question_args)
        if question.strip()
    ]
    if question_file is None:
        if not records:
            raise ValueError("Provide at least one --question or a --question-file.")
        return records

    suffix = question_file.suffix.lower()
    if suffix in {".txt", ".md"}:
        with question_file.open() as fh:
            records.extend({"question": line.strip(), "choices": ""} for line in fh if line.strip())
    elif suffix == ".csv":
        with question_file.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if "question" not in reader.fieldnames:
                raise ValueError("CSV question file must contain a 'question' column.")
            records.extend(
                {
                    "question": row["question"].strip(),
                    "choices": row.get("choices", "").strip(),
                }
                for row in reader
                if row["question"].strip()
            )
    elif suffix == ".json":
        with question_file.open() as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    records.append({"question": item.strip(), "choices": ""})
                elif isinstance(item, dict) and "question" in item:
                    records.append(
                        {
                            "question": str(item["question"]).strip(),
                            "choices": json.dumps(item.get("choices", ""), ensure_ascii=False)
                            if isinstance(item.get("choices", ""), list)
                            else str(item.get("choices", "")),
                        }
                    )
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
                    records.append({"question": item.strip(), "choices": ""})
                elif isinstance(item, dict) and "question" in item:
                    records.append(
                        {
                            "question": str(item["question"]).strip(),
                            "choices": json.dumps(item.get("choices", ""), ensure_ascii=False)
                            if isinstance(item.get("choices", ""), list)
                            else str(item.get("choices", "")),
                        }
                    )
                else:
                    raise ValueError("JSONL items must be strings or dicts with 'question'.")
    else:
        raise ValueError(f"Unsupported question file format: {question_file}")

    records = [record for record in records if record["question"]]
    if not records:
        raise ValueError("No valid questions found.")
    return records


def route_question(bundle: Dict[str, object], question: str, choices: str = "") -> Dict[str, object]:
    pipeline = bundle["pipeline"]
    threshold = float(bundle["min_confidence"])
    default_fs = str(bundle["default_fs"])
    input_mode = str(bundle.get("input_mode", "question"))
    feature_column = str(bundle.get("feature_column", "question"))
    if feature_column == "router_text":
        frame = pd.DataFrame({"router_text": [build_router_text(question, choices, input_mode)]})
    else:
        frame = pd.DataFrame({"question": [question]})
    proba = pipeline.predict_proba(frame)[0]
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
        "choices": choices,
        "input_mode": input_mode,
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
    records = load_records(args.question, args.choices, args.question_file)
    rows = [route_question(bundle, record["question"], record.get("choices", "")) for record in records]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    if args.save_predictions is not None:
        save_predictions(args.save_predictions, rows)


if __name__ == "__main__":
    main()
