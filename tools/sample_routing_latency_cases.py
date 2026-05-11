#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


CASE_HIGH = "case1_high_confidence"
CASE_AGREE = "case2_low_confidence_agree"
CASE_DISAGREE = "case3_low_confidence_disagree"
CASES = [CASE_HIGH, CASE_AGREE, CASE_DISAGREE]


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def classify_case(row: dict) -> str:
    decision = row.get("decision", "")
    low_conf = truthy(row.get("default_low_confidence", row.get("fs224_low_confidence", "")))
    agree = truthy(row.get("default_challenger_agree", row.get("fs112_fs224_agree", "")))

    if decision.startswith("accept_fs") and decision.endswith("_high_confidence"):
        return CASE_HIGH
    if decision == "low_confidence_agree_accept_answer" or (low_conf and agree):
        return CASE_AGREE
    if low_conf and not agree:
        return CASE_DISAGREE
    if "disagreement" in decision:
        return CASE_DISAGREE
    return CASE_HIGH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample routing outputs by latency-relevant cases for cost measurement."
    )
    parser.add_argument("--routed-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--per-case", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="If a case has fewer rows than --per-case, sample all available rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.routed_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty routed CSV: {args.routed_csv}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    by_case = defaultdict(list)
    for idx, row in enumerate(rows):
        row = dict(row)
        row["_routed_row_index"] = str(idx)
        by_case[classify_case(row)].append(row)

    rng = random.Random(args.seed)
    sampled = []
    counts = {case: len(by_case[case]) for case in CASES}
    total = len(rows)

    for case in CASES:
        available = by_case[case]
        if len(available) < args.per_case and not args.allow_short:
            raise ValueError(
                f"{case} has only {len(available)} rows, fewer than --per-case {args.per_case}. "
                "Use --allow-short to sample all available rows."
            )
        chosen = list(available) if len(available) <= args.per_case else rng.sample(available, args.per_case)
        for sample_idx, row in enumerate(chosen):
            out = dict(row)
            out["latency_case"] = case
            out["latency_sample_index"] = str(sample_idx)
            out["case_full_count"] = str(counts[case])
            out["case_full_ratio"] = f"{(counts[case] / total if total else 0.0):.10f}"
            out["case_sample_count"] = str(len(chosen))
            out["full_total_count"] = str(total)
            out["sample_seed"] = str(args.seed)
            sampled.append(out)

    extra_fields = [
        "_routed_row_index",
        "latency_case",
        "latency_sample_index",
        "case_full_count",
        "case_full_ratio",
        "case_sample_count",
        "full_total_count",
        "sample_seed",
    ]
    output_fields = fieldnames + [field for field in extra_fields if field not in fieldnames]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(sampled)

    summary = {
        "routed_csv": str(args.routed_csv),
        "output_csv": str(args.output_csv),
        "total": total,
        "per_case_requested": args.per_case,
        "seed": args.seed,
        "case_counts": counts,
        "case_ratios": {case: (counts[case] / total if total else 0.0) for case in CASES},
        "sample_counts": Counter(row["latency_case"] for row in sampled),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
