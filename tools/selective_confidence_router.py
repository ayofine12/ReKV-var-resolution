#!/usr/bin/env python3
"""
Evaluate confidence-gated fs224 -> fs112 selective routing.

This is an offline evaluator for already-computed result CSVs. It simulates:

  1. Run fs224.
  2. If fs224 confidence is high, accept fs224.
  3. If fs224 confidence is low, run fs112.
  4. If fs112/fs224 disagree, use a cheap meta-verifier that sees only the
     question, choices, candidate answers, and confidence features.

It does not send correct_choice, answer, pred correctness, or qa_acc to the
LLM verifier. Those fields are used only for offline evaluation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


LABEL_112 = "route_112"
LABEL_224 = "route_224"
LABEL_TIE = "tie"

FEATURE_COLUMNS = [
    "top1_prob",
    "top2_prob",
    "prob_margin",
    "logit_margin",
    "choice_entropy",
    "normalized_choice_entropy",
]


@dataclass(frozen=True)
class SelectiveExample:
    index: int
    key: str
    video_id: str
    question: str
    choices: str
    correct_choice: str
    task: str
    row112: Dict[str, str]
    row224: Dict[str, str]
    acc112: float
    acc224: float
    pred112: str
    pred224: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fs224 confidence-gated selective routing with an optional meta-verifier."
    )
    parser.add_argument("--csv-112", nargs="+", required=True, help="One or more fs112 result CSVs.")
    parser.add_argument("--csv-224", nargs="+", required=True, help="One or more fs224 result CSVs.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save per-example routing decisions.")
    parser.add_argument(
        "--gate-column",
        default="prob_margin",
        help="Column in the fs224 CSV used to decide whether fs224 is low-confidence.",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        required=True,
        help="Threshold for --gate-column. Default comparison is value < threshold.",
    )
    parser.add_argument(
        "--low-confidence-when",
        choices=["lt", "le", "gt", "ge"],
        default="lt",
        help="Comparison that marks fs224 as low-confidence. Use lt for margins, gt for entropy.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=FEATURE_COLUMNS,
        help="Confidence feature columns shown to the meta-verifier.",
    )
    parser.add_argument(
        "--verifier",
        choices=["llm", "confidence", "fs224", "fs112", "oracle"],
        default="confidence",
        help=(
            "How to resolve fs112/fs224 disagreement after the fs224 low-confidence gate. "
            "llm calls a prompt-based meta-verifier; confidence compares calibrated-looking margins; "
            "oracle is an upper bound."
        ),
    )
    parser.add_argument(
        "--confidence-compare-column",
        default="prob_margin",
        help="Column compared between fs112 and fs224 when --verifier confidence is used.",
    )
    parser.add_argument(
        "--default-fs",
        choices=["112", "224", "auto"],
        default="224",
        help="Fallback fs for verifier parse failures, ties, or low verifier confidence.",
    )
    parser.add_argument(
        "--include-task",
        action="store_true",
        help="Include task/question_type in the LLM verifier prompt.",
    )
    parser.add_argument("--model", default=os.environ.get("LLM_ROUTER_MODEL"), help="Chat model for --verifier llm.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument(
        "--min-verifier-confidence",
        type=float,
        default=0.0,
        help="If LLM verifier confidence is lower than this value, fall back to --default-fs.",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Seed for randomizing candidate order.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Print one verifier prompt and exit.")
    parser.add_argument("--cost-fs224", type=float, default=1.0)
    parser.add_argument("--cost-fs112", type=float, default=1.0)
    parser.add_argument("--cost-verifier", type=float, default=0.05)
    return parser.parse_args()


def result_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        row["video_id"],
        row["question"],
        row.get("choices", ""),
        row.get("correct_choice", ""),
    )


def stable_key(video_id: str, question: str, choices: str, correct_choice: str) -> str:
    return json.dumps(
        {
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "correct_choice": correct_choice,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def normalize_acc(raw: str) -> float:
    value = float(raw)
    if value > 1.0:
        return value / 100.0
    return value


def parse_float(raw: object, default: float = 0.0) -> float:
    try:
        if raw is None or raw == "":
            return default
        return float(raw)
    except Exception:
        return default


def load_side(paths: Sequence[str], side_name: str) -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    merged: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for path_str in paths:
        path = Path(path_str)
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV for {side_name}: {path}")
            for row in reader:
                key = result_key(row)
                if key in merged:
                    raise ValueError(f"Duplicate key for {side_name}: {key} from {path}")
                row = dict(row)
                row["_source_path"] = str(path)
                merged[key] = row
    return merged


def build_examples(paths112: Sequence[str], paths224: Sequence[str]) -> List[SelectiveExample]:
    rows112 = load_side(paths112, "112")
    rows224 = load_side(paths224, "224")
    common_keys = sorted(set(rows112) & set(rows224))
    if not common_keys:
        raise ValueError("No overlapping questions between the provided fs112 and fs224 CSVs.")

    missing_112 = len(set(rows224) - set(rows112))
    missing_224 = len(set(rows112) - set(rows224))
    if missing_112 or missing_224:
        print(
            f"[warn] matched={len(common_keys)} missing_from_112={missing_112} missing_from_224={missing_224}",
            file=sys.stderr,
        )

    examples: List[SelectiveExample] = []
    for idx, key in enumerate(common_keys):
        row112 = rows112[key]
        row224 = rows224[key]
        examples.append(
            SelectiveExample(
                index=idx,
                key=stable_key(row112["video_id"], row112["question"], row112.get("choices", ""), row112.get("correct_choice", "")),
                video_id=row112["video_id"],
                question=row112["question"],
                choices=row112.get("choices", ""),
                correct_choice=row112.get("correct_choice", ""),
                task=row112.get("task", ""),
                row112=row112,
                row224=row224,
                acc112=normalize_acc(row112["qa_acc"]),
                acc224=normalize_acc(row224["qa_acc"]),
                pred112=row112.get("pred_choice", ""),
                pred224=row224.get("pred_choice", ""),
            )
        )
    return examples


def choose_default_fs(examples: Sequence[SelectiveExample], default_fs: str) -> str:
    if default_fs in {"112", "224"}:
        return default_fs
    mean112 = sum(ex.acc112 for ex in examples) / len(examples)
    mean224 = sum(ex.acc224 for ex in examples) / len(examples)
    return "112" if mean112 >= mean224 else "224"


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


def answer_with_text(letter: str, raw_choices: str) -> str:
    letter = (letter or "").strip().upper()
    choices = parse_choices(raw_choices)
    if len(letter) == 1 and "A" <= letter <= "Z":
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return f"({letter}) {choices[idx]}"
    return letter or "<empty>"


def is_low_confidence(value: float, threshold: float, mode: str) -> bool:
    if mode == "lt":
        return value < threshold
    if mode == "le":
        return value <= threshold
    if mode == "gt":
        return value > threshold
    if mode == "ge":
        return value >= threshold
    raise ValueError(f"Unsupported low-confidence comparison: {mode}")


def gold_label(ex: SelectiveExample) -> str:
    if ex.acc112 > ex.acc224:
        return LABEL_112
    if ex.acc224 > ex.acc112:
        return LABEL_224
    return LABEL_TIE


def fs_to_label(fs: str) -> str:
    if fs == "112":
        return LABEL_112
    if fs == "224":
        return LABEL_224
    return LABEL_TIE


def label_to_fs(label: str, default_fs: str) -> str:
    if label == LABEL_112:
        return "112"
    if label == LABEL_224:
        return "224"
    return default_fs


def acc_for_fs(ex: SelectiveExample, fs: str) -> float:
    return ex.acc112 if fs == "112" else ex.acc224


def feature_block(row: Dict[str, str], columns: Sequence[str]) -> str:
    lines = []
    for column in columns:
        value = row.get(column, "")
        if value != "":
            lines.append(f"- {column}: {value}")
    if not lines:
        return "- no confidence features available"
    return "\n".join(lines)


def randomized_candidates(ex: SelectiveExample, seed: int) -> List[Tuple[str, str, str, Dict[str, str]]]:
    candidates = [
        ("fs224", "224", ex.pred224, ex.row224),
        ("fs112", "112", ex.pred112, ex.row112),
    ]
    rng = random.Random(seed + ex.index)
    rng.shuffle(candidates)
    return candidates


def build_verifier_messages(
    ex: SelectiveExample,
    args: argparse.Namespace,
    candidates: Sequence[Tuple[str, str, str, Dict[str, str]]],
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a cheap meta-verifier for multiple-choice video QA.\n"
        "\n"
        "You do not see video frames. You only see the question, options, two candidate answers, "
        "and confidence features from two resolution runs. Choose which candidate is more likely "
        "to be correct. Do not prefer a candidate because of its order or because of its hidden "
        "resolution. Treat confidence margins as useful but imperfect evidence.\n"
        "\n"
        "Return only JSON with keys choice, confidence, and reason. choice must be \"X\" or \"Y\". "
        "confidence must be a number from 0 to 1. Keep reason short."
    )

    labels = ["X", "Y"]
    candidate_lines = []
    for visible_label, (_, _, pred, row) in zip(labels, candidates):
        candidate_lines.append(
            f"Candidate {visible_label}\n"
            f"answer: {answer_with_text(pred, ex.choices)}\n"
            f"confidence features:\n{feature_block(row, args.feature_columns)}"
        )

    parts = [
        f"Question:\n{ex.question}",
        f"Options:\n{format_choices(ex.choices)}",
    ]
    if args.include_task and ex.task:
        parts.append(f"Task/question type:\n{ex.task}")
    parts.extend(candidate_lines)
    user_prompt = "\n\n".join(parts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_openai_client(api_key_env: str, base_url: str | None, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for --verifier llm.") from exc

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")

    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def token_limit_kwargs(model: str, max_tokens: int) -> Dict[str, int]:
    model_name = model.lower()
    if model_name.startswith("gpt-5") or model_name.startswith("o1") or model_name.startswith("o3"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def request_llm(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format_json: bool,
    max_retries: int,
    retry_sleep: float,
) -> Tuple[str, str | None]:
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            kwargs.update(token_limit_kwargs(model, max_tokens))
            if response_format_json:
                kwargs["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            return content or "", None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2**attempt))
    return "", last_error


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_verifier_json(content: str) -> Tuple[str, float, str, str | None]:
    text = strip_code_fence(content)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except Exception as exc:
        return "", 0.0, "", f"json_parse_error: {exc}"

    choice = str(data.get("choice", "")).strip().upper()
    if choice not in {"X", "Y"}:
        return "", 0.0, str(data.get("reason", "")), f"bad_choice: {choice}"
    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return choice, confidence, str(data.get("reason", "")), None


def choose_by_confidence(ex: SelectiveExample, column: str, default_fs: str) -> Tuple[str, float, str]:
    score112 = parse_float(ex.row112.get(column), default=0.0)
    score224 = parse_float(ex.row224.get(column), default=0.0)
    if score112 > score224:
        return "112", max(score112, score224), f"fs112 {column} higher"
    if score224 > score112:
        return "224", max(score112, score224), f"fs224 {column} higher"
    return default_fs, score224, f"{column} tie; fallback {default_fs}"


def choose_with_oracle(ex: SelectiveExample, default_fs: str) -> Tuple[str, str]:
    label = gold_label(ex)
    if label == LABEL_112:
        return "112", "oracle chose fs112"
    if label == LABEL_224:
        return "224", "oracle chose fs224"
    return default_fs, f"oracle tie; fallback {default_fs}"


def route_one(
    ex: SelectiveExample,
    args: argparse.Namespace,
    default_fs: str,
    client: Any | None,
) -> Dict[str, object]:
    gate_value = parse_float(ex.row224.get(args.gate_column), default=float("nan"))
    if gate_value != gate_value:
        raise ValueError(
            f"Missing or non-numeric fs224 gate column {args.gate_column!r}. "
            "Run fs224 with --save_choice_scores True first."
        )
    low_conf = is_low_confidence(gate_value, args.gate_threshold, args.low_confidence_when)
    gold = gold_label(ex)
    selective_oracle_fs = "224" if not low_conf else ("112" if ex.acc112 > ex.acc224 else "224")

    base_row: Dict[str, object] = {
        "example_index": ex.index,
        "key": ex.key,
        "video_id": ex.video_id,
        "question": ex.question,
        "choices": ex.choices,
        "task": ex.task,
        "correct_choice": ex.correct_choice,
        "pred112": ex.pred112,
        "pred224": ex.pred224,
        "acc112": ex.acc112,
        "acc224": ex.acc224,
        "gold_label": gold,
        "gate_column": args.gate_column,
        "gate_threshold": args.gate_threshold,
        "low_confidence_when": args.low_confidence_when,
        "fs224_gate_value": gate_value,
        "fs224_low_confidence": low_conf,
        "fs112_fs224_agree": ex.pred112 == ex.pred224 and bool(ex.pred112),
        "verifier": args.verifier,
        "verifier_choice": "",
        "verifier_confidence": "",
        "verifier_reason": "",
        "parse_error": "",
        "raw_response": "",
        "selective_oracle_fs": selective_oracle_fs,
        "selective_oracle_acc": acc_for_fs(ex, selective_oracle_fs),
        "full_oracle_acc": max(ex.acc112, ex.acc224),
        "source_112": ex.row112.get("_source_path", ""),
        "source_224": ex.row224.get("_source_path", ""),
    }

    if not low_conf:
        routed_fs = "224"
        pred_label = LABEL_224
        decision = "accept_fs224_high_confidence"
    elif ex.pred112 == ex.pred224 and ex.pred112:
        routed_fs = "224"
        pred_label = LABEL_224
        decision = "low_confidence_agree_accept_answer"
    elif args.verifier == "fs224":
        routed_fs = "224"
        pred_label = LABEL_224
        decision = "verifier_fallback_fs224"
    elif args.verifier == "fs112":
        routed_fs = "112"
        pred_label = LABEL_112
        decision = "verifier_fallback_fs112"
    elif args.verifier == "oracle":
        routed_fs, reason = choose_with_oracle(ex, default_fs)
        pred_label = fs_to_label(routed_fs)
        decision = "oracle_disagreement"
        base_row["verifier_reason"] = reason
    elif args.verifier == "confidence":
        routed_fs, confidence, reason = choose_by_confidence(ex, args.confidence_compare_column, default_fs)
        pred_label = fs_to_label(routed_fs)
        decision = "confidence_disagreement"
        base_row["verifier_confidence"] = confidence
        base_row["verifier_reason"] = reason
    elif args.verifier == "llm":
        if client is None:
            raise RuntimeError("--verifier llm requires an OpenAI-compatible client.")
        candidates = randomized_candidates(ex, args.seed)
        messages = build_verifier_messages(ex, args, candidates)
        raw_response, request_error = request_llm(
            client=client,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            response_format_json=args.response_format_json,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        base_row["raw_response"] = raw_response
        if request_error is not None:
            choice = ""
            verifier_confidence = 0.0
            reason = ""
            parse_error = request_error
        else:
            choice, verifier_confidence, reason, parse_error = parse_verifier_json(raw_response)
        if choice in {"X", "Y"}:
            chosen_idx = 0 if choice == "X" else 1
            _, routed_fs, _, _ = candidates[chosen_idx]
            pred_label = fs_to_label(routed_fs)
        else:
            routed_fs = default_fs
            pred_label = LABEL_TIE
        if verifier_confidence < args.min_verifier_confidence:
            routed_fs = default_fs
            pred_label = LABEL_TIE
            parse_error = parse_error or f"verifier_confidence_below_threshold: {verifier_confidence:.3f}"
        decision = "llm_verifier_disagreement"
        base_row["verifier_choice"] = choice
        base_row["verifier_confidence"] = verifier_confidence
        base_row["verifier_reason"] = reason
        base_row["parse_error"] = parse_error or ""
    else:
        raise ValueError(f"Unsupported verifier: {args.verifier}")

    routed_acc = acc_for_fs(ex, routed_fs)
    base_row.update(
        {
            "decision": decision,
            "pred_label": pred_label,
            "default_fs": default_fs,
            "routed_fs": routed_fs,
            "routed_acc": routed_acc,
        }
    )
    return base_row


def output_fieldnames() -> List[str]:
    return [
        "example_index",
        "key",
        "video_id",
        "question",
        "choices",
        "task",
        "correct_choice",
        "pred112",
        "pred224",
        "acc112",
        "acc224",
        "gold_label",
        "gate_column",
        "gate_threshold",
        "low_confidence_when",
        "fs224_gate_value",
        "fs224_low_confidence",
        "fs112_fs224_agree",
        "verifier",
        "decision",
        "pred_label",
        "verifier_choice",
        "verifier_confidence",
        "verifier_reason",
        "parse_error",
        "default_fs",
        "routed_fs",
        "routed_acc",
        "selective_oracle_fs",
        "selective_oracle_acc",
        "full_oracle_acc",
        "source_112",
        "source_224",
        "raw_response",
    ]


def save_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fieldnames(), extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["example_index"])):
            writer.writerow(row)


def load_existing_rows(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, object]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = row.get("key", "")
            if key:
                rows[key] = dict(row)
    return rows


def select_examples(examples: Sequence[SelectiveExample], start: int, limit: int | None) -> List[SelectiveExample]:
    if start < 0:
        raise ValueError("--start must be non-negative.")
    selected = list(examples[start:])
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        selected = selected[:limit]
    return selected


def rows_from_existing(
    examples: Sequence[SelectiveExample],
    existing_by_key: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[SelectiveExample]]:
    rows: List[Dict[str, object]] = []
    remaining: List[SelectiveExample] = []
    for ex in examples:
        existing = existing_by_key.get(ex.key)
        if existing is None:
            remaining.append(ex)
        else:
            rows.append(existing)
    return rows, remaining


def coerce_float(value: object) -> float:
    return parse_float(value, default=0.0)


def summarize(rows: Sequence[Dict[str, object]], args: argparse.Namespace, default_fs: str) -> None:
    if not rows:
        print("[summary]\nn_examples: 0")
        return

    acc112 = [coerce_float(row["acc112"]) for row in rows]
    acc224 = [coerce_float(row["acc224"]) for row in rows]
    routed = [coerce_float(row["routed_acc"]) for row in rows]
    selective_oracle = [coerce_float(row["selective_oracle_acc"]) for row in rows]
    full_oracle = [coerce_float(row["full_oracle_acc"]) for row in rows]
    base112 = sum(acc112) / len(rows)
    base224 = sum(acc224) / len(rows)
    default_acc = base112 if default_fs == "112" else base224
    routed_acc = sum(routed) / len(rows)
    low_conf_rows = [row for row in rows if str(row["fs224_low_confidence"]) == "True"]
    verifier_rows = [
        row
        for row in low_conf_rows
        if str(row.get("fs112_fs224_agree", "")) != "True"
    ]
    parse_errors = sum(1 for row in rows if str(row.get("parse_error", "")))

    decisive_verifier = [row for row in verifier_rows if row["gold_label"] != LABEL_TIE]
    verifier_label_acc = (
        sum(1 for row in decisive_verifier if row["pred_label"] == row["gold_label"]) / len(decisive_verifier)
        if decisive_verifier
        else 0.0
    )

    avg_cost = (
        args.cost_fs224
        + (len(low_conf_rows) / len(rows)) * args.cost_fs112
        + (len(verifier_rows) / len(rows)) * args.cost_verifier
    )
    always_both_cost = args.cost_fs224 + args.cost_fs112 + args.cost_verifier

    print("[summary]")
    print(f"n_examples: {len(rows)}")
    print(f"default_fs: {default_fs}")
    print(f"gate: fs224.{args.gate_column} {args.low_confidence_when} {args.gate_threshold}")
    print(f"verifier: {args.verifier}")
    print(f"base_acc_112: {100 * base112:.2f}")
    print(f"base_acc_224: {100 * base224:.2f}")
    print(f"default_acc: {100 * default_acc:.2f}")
    print(f"routed_acc: {100 * routed_acc:.2f}")
    print(f"gain_vs_default: {100 * (routed_acc - default_acc):.2f}p")
    print(f"gain_vs_112: {100 * (routed_acc - base112):.2f}p")
    print(f"gain_vs_224: {100 * (routed_acc - base224):.2f}p")
    print(f"selective_oracle_acc: {100 * (sum(selective_oracle) / len(rows)):.2f}")
    print(f"full_oracle_acc: {100 * (sum(full_oracle) / len(rows)):.2f}")
    print(f"low_confidence_ratio: {100 * (len(low_conf_rows) / len(rows)):.2f}")
    print(f"verifier_call_ratio: {100 * (len(verifier_rows) / len(rows)):.2f}")
    print(f"verifier_decisive_n: {len(decisive_verifier)}")
    print(f"verifier_decisive_label_acc: {100 * verifier_label_acc:.2f}")
    print(f"parse_error_count: {parse_errors}")
    print(f"avg_relative_cost: {avg_cost:.4f}")
    print(f"always_both_relative_cost: {always_both_cost:.4f}")
    print(f"cost_vs_always_both: {100 * (avg_cost / always_both_cost):.2f}%")
    print(f"decision_distribution: {json.dumps(Counter(str(row['decision']) for row in rows), ensure_ascii=False)}")
    print(f"gold_distribution: {json.dumps(Counter(str(row['gold_label']) for row in rows), ensure_ascii=False)}")
    print(f"prediction_distribution: {json.dumps(Counter(str(row['pred_label']) for row in rows), ensure_ascii=False)}")


def print_dry_run(examples: Sequence[SelectiveExample], args: argparse.Namespace, default_fs: str) -> None:
    selected = None
    for ex in examples:
        gate_value = parse_float(ex.row224.get(args.gate_column), default=float("nan"))
        if gate_value == gate_value and is_low_confidence(gate_value, args.gate_threshold, args.low_confidence_when):
            if ex.pred112 != ex.pred224:
                selected = ex
                break
    if selected is None:
        selected = examples[0]
    candidates = randomized_candidates(selected, args.seed)
    messages = build_verifier_messages(selected, args, candidates)
    print("[dry_run_messages]")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print(f"[dry_run] selected_examples={len(examples)} default_fs={default_fs}")


def route_remaining(
    remaining: Sequence[SelectiveExample],
    args: argparse.Namespace,
    default_fs: str,
    client: Any | None,
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not remaining:
        return rows

    completed_since_flush = 0
    if args.workers <= 1 or args.verifier != "llm":
        for position, ex in enumerate(remaining, start=1):
            row = route_one(ex, args, default_fs, client)
            rows.append(row)
            completed_since_flush += 1
            print(
                f"[routed] {position}/{len(remaining)} idx={ex.index} "
                f"decision={row['decision']} routed={row['routed_fs']} "
                f"acc={float(row['routed_acc']):.2f}",
                flush=True,
            )
            if args.flush_every > 0 and completed_since_flush >= args.flush_every:
                save_rows(args.output, rows)
                completed_since_flush = 0
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(route_one, ex, args, default_fs, client): ex
                for ex in remaining
            }
            for position, future in enumerate(as_completed(futures), start=1):
                ex = futures[future]
                row = future.result()
                rows.append(row)
                completed_since_flush += 1
                print(
                    f"[routed] {position}/{len(remaining)} idx={ex.index} "
                    f"decision={row['decision']} routed={row['routed_fs']} "
                    f"acc={float(row['routed_acc']):.2f}",
                    flush=True,
                )
                if args.flush_every > 0 and completed_since_flush >= args.flush_every:
                    save_rows(args.output, rows)
                    completed_since_flush = 0
    return rows


def main() -> None:
    args = parse_args()
    if args.verifier == "llm" and not args.model and not args.dry_run:
        raise ValueError("Provide --model or set LLM_ROUTER_MODEL for --verifier llm.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    examples = build_examples(args.csv_112, args.csv_224)
    default_fs = choose_default_fs(examples, args.default_fs)
    selected_examples = select_examples(examples, args.start, args.limit)
    if not selected_examples:
        raise ValueError("No examples selected.")

    if args.dry_run:
        print_dry_run(selected_examples, args, default_fs)
        return

    existing = load_existing_rows(args.output) if args.resume else {}
    rows, remaining = rows_from_existing(selected_examples, existing)
    print(f"[setup] joined_examples={len(examples)} selected={len(selected_examples)}")
    print(f"[setup] reused={len(rows)} remaining={len(remaining)}")
    print(f"[setup] default_fs={default_fs} verifier={args.verifier}")

    client = None
    if args.verifier == "llm":
        client = load_openai_client(args.api_key_env, args.base_url, args.timeout)

    rows = route_remaining(remaining, args, default_fs, client, rows)
    save_rows(args.output, rows)
    summarize(rows, args, default_fs)
    print(f"[saved_predictions] {args.output}")


if __name__ == "__main__":
    main()
