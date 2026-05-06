#!/usr/bin/env python3
"""
Route each video QA question to fs112 or fs224 with a text-only LLM.

The router can run in two input modes:
  - question: only the question text is shown to the LLM.
  - question_choices: the question and multiple-choice options are shown.

The script joins existing fs112/fs224 result CSVs, asks an OpenAI-compatible
chat model to choose a route, and evaluates the routed accuracy from the
already-computed qa_acc columns. It never sends task/question_type, answer,
correct_choice, pred_choice, or qa_acc to the LLM.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


LABEL_112 = "route_112"
LABEL_224 = "route_224"
LABEL_TIE = "tie"

ROUTE_TO_LABEL = {
    "112": LABEL_112,
    "fs112": LABEL_112,
    "route_112": LABEL_112,
    "224": LABEL_224,
    "fs224": LABEL_224,
    "route_224": LABEL_224,
}


@dataclass(frozen=True)
class RouterExample:
    index: int
    video_id: str
    question: str
    choices: str
    correct_choice: str
    acc112: float
    acc224: float
    gold_label: str
    source_112: str
    source_224: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM fs112/fs224 resolution router from result CSVs."
    )
    parser.add_argument("--csv-112", nargs="+", required=True, help="One or more fs112 result CSVs.")
    parser.add_argument("--csv-224", nargs="+", required=True, help="One or more fs224 result CSVs.")
    parser.add_argument(
        "--input-mode",
        choices=["question", "question_choices"],
        required=True,
        help="Whether the LLM sees only the question or the question plus choices.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save per-question routing predictions as CSV.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_ROUTER_MODEL"),
        help="Chat model name. Can also be set with LLM_ROUTER_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per request after failures.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Initial retry sleep in seconds.")
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Optional throttle for sequential mode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel request workers. Keep at 1 for strict rate control.",
    )
    parser.add_argument(
        "--default-fs",
        choices=["112", "224", "auto"],
        default="auto",
        help="Fallback fs for parse failures or confidence abstention.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="If the LLM confidence is below this value, abstain to the default fs.",
    )
    parser.add_argument(
        "--response-format-json",
        action="store_true",
        help="Pass response_format={type: json_object}. Disable for endpoints that do not support it.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Route at most this many examples.")
    parser.add_argument("--start", type=int, default=0, help="Start offset after joining examples.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse rows already present in --output for the same input mode.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Write partial results after this many new predictions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt and exit without calling the LLM.",
    )
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


def choose_gold_label(acc112: float, acc224: float) -> str:
    if acc112 > acc224:
        return LABEL_112
    if acc224 > acc112:
        return LABEL_224
    return LABEL_TIE


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

    missing_112 = len(set(rows224) - set(rows112))
    missing_224 = len(set(rows112) - set(rows224))
    if missing_112 or missing_224:
        print(
            f"[warn] matched={len(common_keys)} missing_from_112={missing_112} missing_from_224={missing_224}",
            file=sys.stderr,
        )

    examples: List[RouterExample] = []
    for idx, key in enumerate(common_keys):
        r112 = rows112[key]
        r224 = rows224[key]
        acc112 = normalize_acc(r112["qa_acc"])
        acc224 = normalize_acc(r224["qa_acc"])
        examples.append(
            RouterExample(
                index=idx,
                video_id=r112["video_id"],
                question=r112["question"],
                choices=r112.get("choices", ""),
                correct_choice=r112.get("correct_choice", ""),
                acc112=acc112,
                acc224=acc224,
                gold_label=choose_gold_label(acc112, acc224),
                source_112=r112.get("_source_path", ""),
                source_224=r224.get("_source_path", ""),
            )
        )
    return examples


def choose_default_fs(examples: Sequence[RouterExample], default_fs: str) -> str:
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
    choices = parse_choices(raw)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for idx, choice in enumerate(choices):
        prefix = labels[idx] if idx < len(labels) else str(idx + 1)
        lines.append(f"({prefix}) {choice}")
    return "\n".join(lines)


def build_messages(example: RouterExample, input_mode: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You route a multiple-choice video QA request to one of two video preprocessing settings.\n"
        "\n"
        "fs112: lower spatial resolution, wider temporal coverage. Prefer it when the question "
        "seems to require temporal order, long-range context, events before/after, repeated "
        "actions, or overall story understanding.\n"
        "\n"
        "fs224: higher spatial resolution, narrower temporal coverage. Prefer it when the question "
        "seems to require fine visual details, small objects, text/signs/logos, colors, clothing, "
        "object identity, person/team/brand recognition, or spatial attributes.\n"
        "\n"
        "Use only the given question text and, if present, choices. Do not answer the QA question. "
        "Choose the route that is more likely to let a video QA model answer correctly. "
        "Output only a JSON object with keys route, confidence, and reason. route must be "
        "either \"fs112\" or \"fs224\". confidence must be a number from 0 to 1. Keep reason short."
    )
    if input_mode == "question":
        user_prompt = f"Question:\n{example.question}"
    elif input_mode == "question_choices":
        user_prompt = f"Question:\n{example.question}\n\nChoices:\n{format_choices(example.choices)}"
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_openai_client(api_key_env: str, base_url: str | None, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required. Install project dependencies first.") from exc

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")

    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_llm_json(content: str) -> Tuple[str, float, str, str | None]:
    text = strip_code_fence(content)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except Exception as exc:
        return LABEL_TIE, 0.0, "", f"json_parse_error: {exc}"

    route_raw = str(data.get("route", "")).strip().lower().replace("-", "_")
    pred_label = ROUTE_TO_LABEL.get(route_raw)
    if pred_label is None:
        return LABEL_TIE, 0.0, str(data.get("reason", "")), f"bad_route: {route_raw}"

    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", ""))
    return pred_label, confidence, reason, None


def token_limit_kwargs(model: str, max_tokens: int) -> Dict[str, int]:
    model_name = model.lower()
    if model_name.startswith("gpt-5") or model_name.startswith("o1") or model_name.startswith("o3"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def request_route(
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


def routed_fs_from_label(pred_label: str, default_fs: str) -> str:
    if pred_label == LABEL_112:
        return "112"
    if pred_label == LABEL_224:
        return "224"
    return default_fs


def routed_acc_from_label(pred_label: str, example: RouterExample, default_fs: str) -> float:
    routed_fs = routed_fs_from_label(pred_label, default_fs)
    return example.acc112 if routed_fs == "112" else example.acc224


def route_one(
    example: RouterExample,
    args: argparse.Namespace,
    client: Any,
    default_fs: str,
) -> Dict[str, object]:
    messages = build_messages(example, args.input_mode)
    raw_content, request_error = request_route(
        client=client,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        response_format_json=args.response_format_json,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    if request_error is not None:
        pred_label = LABEL_TIE
        confidence = 0.0
        reason = ""
        parse_error = request_error
    else:
        pred_label, confidence, reason, parse_error = parse_llm_json(raw_content)

    raw_pred_label = pred_label
    if pred_label != LABEL_TIE and confidence < args.min_confidence:
        pred_label = LABEL_TIE
        parse_error = parse_error or f"confidence_below_threshold: {confidence:.3f}"

    routed_fs = routed_fs_from_label(pred_label, default_fs)
    routed_acc = routed_acc_from_label(pred_label, example, default_fs)
    key = stable_key(example.video_id, example.question, example.choices, example.correct_choice)
    return {
        "example_index": example.index,
        "input_mode": args.input_mode,
        "model": args.model,
        "video_id": example.video_id,
        "question": example.question,
        "choices": example.choices,
        "correct_choice": example.correct_choice,
        "key": key,
        "acc112": example.acc112,
        "acc224": example.acc224,
        "gold_label": example.gold_label,
        "raw_pred_label": raw_pred_label,
        "pred_label": pred_label,
        "confidence": confidence,
        "reason": reason,
        "parse_error": parse_error or "",
        "default_fs": default_fs,
        "routed_fs": routed_fs,
        "routed_acc": routed_acc,
        "source_112": example.source_112,
        "source_224": example.source_224,
        "raw_response": raw_content,
    }


def output_fieldnames() -> List[str]:
    return [
        "example_index",
        "input_mode",
        "model",
        "video_id",
        "question",
        "choices",
        "correct_choice",
        "key",
        "acc112",
        "acc224",
        "gold_label",
        "raw_pred_label",
        "pred_label",
        "confidence",
        "reason",
        "parse_error",
        "default_fs",
        "routed_fs",
        "routed_acc",
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


def load_existing_rows(path: Path, input_mode: str) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, object]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("input_mode") != input_mode:
                continue
            parse_error = row.get("parse_error", "")
            if parse_error and not parse_error.startswith("confidence_below_threshold"):
                continue
            key = row.get("key", "")
            if key:
                rows[key] = dict(row)
    return rows


def select_examples(examples: Sequence[RouterExample], start: int, limit: int | None) -> List[RouterExample]:
    if start < 0:
        raise ValueError("--start must be non-negative.")
    selected = list(examples[start:])
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        selected = selected[:limit]
    return selected


def coerce_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def summarize(rows: Sequence[Dict[str, object]], default_fs: str) -> None:
    if not rows:
        print("[summary]\nn_examples: 0")
        return

    acc112 = [coerce_float(row["acc112"]) for row in rows]
    acc224 = [coerce_float(row["acc224"]) for row in rows]
    routed_accs = [coerce_float(row["routed_acc"]) for row in rows]
    oracle_accs = [max(a112, a224) for a112, a224 in zip(acc112, acc224)]
    base112 = sum(acc112) / len(acc112)
    base224 = sum(acc224) / len(acc224)
    default_acc = base112 if default_fs == "112" else base224
    routed_acc = sum(routed_accs) / len(routed_accs)
    oracle_acc = sum(oracle_accs) / len(oracle_accs)

    gold_counter = Counter(str(row["gold_label"]) for row in rows)
    pred_counter = Counter(str(row["pred_label"]) for row in rows)
    parse_error_count = sum(1 for row in rows if str(row.get("parse_error", "")))

    decisive = [row for row in rows if row["gold_label"] != LABEL_TIE]
    decisive_correct = sum(1 for row in decisive if row["pred_label"] == row["gold_label"])
    decisive_acc = decisive_correct / len(decisive) if decisive else 0.0

    switch_count = sum(1 for row in rows if row["pred_label"] != LABEL_TIE)
    switch_ratio = switch_count / len(rows)

    print("[summary]")
    print(f"n_examples: {len(rows)}")
    print(f"default_fs: {default_fs}")
    print(f"base_acc_112: {100 * base112:.2f}")
    print(f"base_acc_224: {100 * base224:.2f}")
    print(f"default_acc: {100 * default_acc:.2f}")
    print(f"question_oracle_acc: {100 * oracle_acc:.2f}")
    print(f"llm_routed_acc: {100 * routed_acc:.2f}")
    print(f"gain_vs_default: {100 * (routed_acc - default_acc):.2f}p")
    print(f"gain_vs_112: {100 * (routed_acc - base112):.2f}p")
    print(f"gain_vs_224: {100 * (routed_acc - base224):.2f}p")
    print(f"switch_ratio: {100 * switch_ratio:.2f}")
    print(f"decisive_n: {len(decisive)}")
    print(f"decisive_router_acc: {100 * decisive_acc:.2f}")
    print(f"parse_error_count: {parse_error_count}")
    print(f"gold_distribution: {json.dumps(gold_counter, ensure_ascii=False)}")
    print(f"prediction_distribution: {json.dumps(pred_counter, ensure_ascii=False)}")


def print_dry_run_prompt(example: RouterExample, input_mode: str) -> None:
    messages = build_messages(example, input_mode)
    print("[dry_run_messages]")
    print(json.dumps(messages, ensure_ascii=False, indent=2))


def rows_from_existing(
    selected_examples: Sequence[RouterExample],
    existing_by_key: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[RouterExample]]:
    rows: List[Dict[str, object]] = []
    remaining: List[RouterExample] = []
    for ex in selected_examples:
        key = stable_key(ex.video_id, ex.question, ex.choices, ex.correct_choice)
        existing = existing_by_key.get(key)
        if existing is None:
            remaining.append(ex)
        else:
            rows.append(existing)
    return rows, remaining


def route_remaining(
    remaining: Sequence[RouterExample],
    args: argparse.Namespace,
    client: Any,
    default_fs: str,
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not remaining:
        return rows

    completed_since_flush = 0
    if args.workers <= 1:
        for position, example in enumerate(remaining, start=1):
            row = route_one(example, args, client, default_fs)
            rows.append(row)
            completed_since_flush += 1
            print(
                f"[routed] {position}/{len(remaining)} idx={example.index} "
                f"pred={row['pred_label']} conf={float(row['confidence']):.2f} "
                f"gold={example.gold_label} acc={float(row['routed_acc']):.2f}",
                flush=True,
            )
            if args.flush_every > 0 and completed_since_flush >= args.flush_every:
                save_rows(args.output, rows)
                completed_since_flush = 0
            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(route_one, example, args, client, default_fs): example
                for example in remaining
            }
            for position, future in enumerate(as_completed(futures), start=1):
                example = futures[future]
                row = future.result()
                rows.append(row)
                completed_since_flush += 1
                print(
                    f"[routed] {position}/{len(remaining)} idx={example.index} "
                    f"pred={row['pred_label']} conf={float(row['confidence']):.2f} "
                    f"gold={example.gold_label} acc={float(row['routed_acc']):.2f}",
                    flush=True,
                )
                if args.flush_every > 0 and completed_since_flush >= args.flush_every:
                    save_rows(args.output, rows)
                    completed_since_flush = 0

    return rows


def main() -> None:
    args = parse_args()
    if not args.model and not args.dry_run:
        raise ValueError("Provide --model or set LLM_ROUTER_MODEL.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    examples = build_examples(args.csv_112, args.csv_224)
    default_fs = choose_default_fs(examples, args.default_fs)
    selected_examples = select_examples(examples, args.start, args.limit)
    if not selected_examples:
        raise ValueError("No examples selected.")

    if args.dry_run:
        print_dry_run_prompt(selected_examples[0], args.input_mode)
        print(f"[dry_run] selected_examples={len(selected_examples)} default_fs={default_fs}")
        return

    existing_by_key = load_existing_rows(args.output, args.input_mode) if args.resume else {}
    rows, remaining = rows_from_existing(selected_examples, existing_by_key)
    print(f"[setup] joined_examples={len(examples)} selected={len(selected_examples)}")
    print(f"[setup] reused={len(rows)} remaining={len(remaining)} input_mode={args.input_mode}")
    print(f"[setup] default_fs={default_fs} model={args.model}")

    client = load_openai_client(args.api_key_env, args.base_url, args.timeout)
    rows = route_remaining(remaining, args, client, default_fs, rows)
    save_rows(args.output, rows)
    summarize(rows, default_fs)
    print(f"[saved_predictions] {args.output}")


if __name__ == "__main__":
    main()
