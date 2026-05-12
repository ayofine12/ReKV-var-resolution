#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize unique-video cache build latency for fs112/fs224 from routing latency CSV."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def numeric(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_ms": mean(values),
        "median_ms": median(values),
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)

    by_video: dict[str, dict[str, float]] = {}
    for row in rows:
        video_id = row.get("video_id") or row.get("key") or row.get("sample_uid")
        entry = by_video.setdefault(str(video_id), {})
        fs224 = numeric(row.get("fs224_cache_build_ms_for_video"))
        fs112 = numeric(row.get("fs112_cache_build_ms_for_video"))
        if fs224 is not None:
            entry["fs224"] = fs224
        if fs112 is not None:
            entry["fs112"] = fs112

    paired = [
        {
            "video_id": video_id,
            "fs224_ms": values["fs224"],
            "fs112_ms": values["fs112"],
            "both_sequential_ms": values["fs224"] + values["fs112"],
            "extra_vs_fs224_ms": values["fs112"],
            "extra_vs_fs224_ratio": (values["fs224"] + values["fs112"]) / values["fs224"]
            if values["fs224"]
            else 0.0,
            "extra_vs_fs112_ms": values["fs224"],
            "extra_vs_fs112_ratio": (values["fs224"] + values["fs112"]) / values["fs112"]
            if values["fs112"]
            else 0.0,
        }
        for video_id, values in by_video.items()
        if "fs224" in values and "fs112" in values
    ]

    fs224_values = [row["fs224_ms"] for row in paired]
    fs112_values = [row["fs112_ms"] for row in paired]
    both_values = [row["both_sequential_ms"] for row in paired]
    ratio_vs_fs224 = [row["extra_vs_fs224_ratio"] for row in paired]
    ratio_vs_fs112 = [row["extra_vs_fs112_ratio"] for row in paired]

    summary = {
        "input_csv": str(args.input_csv),
        "input_rows": len(rows),
        "unique_videos": len(by_video),
        "paired_unique_videos": len(paired),
        "fs224_cache_build": summarize(fs224_values),
        "fs112_cache_build": summarize(fs112_values),
        "both_sequential_cache_build": summarize(both_values),
        "both_vs_fs224_ratio": summarize(ratio_vs_fs224),
        "both_vs_fs112_ratio": summarize(ratio_vs_fs112),
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")


if __name__ == "__main__":
    main()
