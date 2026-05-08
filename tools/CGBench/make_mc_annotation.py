#!/usr/bin/env python
import argparse
import json
import string
from collections import OrderedDict
from pathlib import Path


CHOICE_LETTERS = string.ascii_uppercase


def normalize_intervals(intervals):
    normalized = []
    for interval in intervals or []:
        if len(interval) != 2:
            continue
        start, end = interval
        normalized.append([float(start), float(end)])
    return normalized


def convert_annotation(raw_anno, base_video_dir, video_ext, limit_questions):
    grouped = OrderedDict()
    question_count = 0

    for sample in raw_anno:
        if limit_questions is not None and question_count >= limit_questions:
            break

        video_uid = sample["video_uid"]
        choices = list(sample["choices"])
        right_answer = str(sample["right_answer"]).strip().upper()
        if right_answer not in CHOICE_LETTERS[: len(choices)]:
            raise ValueError(f"Invalid right_answer={right_answer!r} for qid={sample.get('qid')}")

        correct_idx = CHOICE_LETTERS.index(right_answer)
        answer = sample.get("answer")
        if answer != choices[correct_idx]:
            raise ValueError(
                f"Answer mismatch for qid={sample.get('qid')}: "
                f"answer={answer!r}, right_answer={right_answer}, choice={choices[correct_idx]!r}"
            )

        item = grouped.setdefault(
            video_uid,
            {
                "video_id": video_uid,
                "video_path": str(base_video_dir / f"{video_uid}{video_ext}"),
                "duration": sample.get("duration"),
                "conversations": [],
            },
        )

        conv = {
            "sample_id": sample.get("qid"),
            "question": sample["question"],
            "choices": choices,
            "answer": answer,
            "question_type": " / ".join(
                part for part in [sample.get("domain"), sample.get("sub_category")] if part
            ),
        }
        temporal_windows = normalize_intervals(sample.get("clue_intervals"))
        if temporal_windows:
            conv["temporal_windows"] = temporal_windows
        item["conversations"].append(conv)
        question_count += 1

    return list(grouped.values()), question_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_src", type=Path, required=True)
    parser.add_argument("--anno_abs", type=Path, required=True)
    parser.add_argument("--base_video_dir", type=Path, required=True)
    parser.add_argument("--video_ext", type=str, default=".mp4")
    parser.add_argument("--limit_questions", type=int, default=None)
    args = parser.parse_args()

    with args.anno_src.open(encoding="utf-8") as fh:
        raw_anno = json.load(fh)

    converted, question_count = convert_annotation(
        raw_anno=raw_anno,
        base_video_dir=args.base_video_dir,
        video_ext=args.video_ext,
        limit_questions=args.limit_questions,
    )

    args.anno_abs.parent.mkdir(parents=True, exist_ok=True)
    with args.anno_abs.open("w", encoding="utf-8") as fh:
        json.dump(converted, fh, ensure_ascii=False)

    print(
        f"normalized {question_count} CG-Bench QA samples into "
        f"{len(converted)} videos: {args.anno_abs}"
    )


if __name__ == "__main__":
    main()

