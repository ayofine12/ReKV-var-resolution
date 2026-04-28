#!/usr/bin/env python3
"""Convert official MLVU JSON files into the ReKV VQA annotation schema."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlvu-root", type=Path, default=Path("/mnt/ssd1/mwnoh/MLVU/MLVU"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/ssd1/mwnoh/MLVU/MLVU/annotations/mlvu_all_rekv.json"),
    )
    parser.add_argument(
        "--include-open-ended",
        action="store_true",
        help="Include sub_scene/summary style open-ended items without choices.",
    )
    args = parser.parse_args()

    json_dir = args.mlvu_root / "json"
    video_dir = args.mlvu_root / "video"
    if not json_dir.is_dir():
        raise SystemExit(f"Missing JSON directory: {json_dir}")
    if not video_dir.is_dir():
        raise SystemExit(f"Missing video directory: {video_dir}")

    samples: OrderedDict[tuple[str, str], dict] = OrderedDict()
    missing_videos: list[str] = []
    skipped_open_ended = 0

    for json_path in sorted(json_dir.glob("*.json")):
        task_folder = json_path.stem
        with json_path.open() as f:
            rows = json.load(f)

        for idx, row in enumerate(rows):
            choices = row.get("candidates")
            if choices is None and not args.include_open_ended:
                skipped_open_ended += 1
                continue

            video_name = row["video"]
            video_path = video_dir / task_folder / video_name
            if not video_path.exists():
                missing_videos.append(str(video_path))

            video_stem = Path(video_name).stem
            key = (task_folder, video_name)
            if key not in samples:
                samples[key] = {
                    "video_id": f"{task_folder}/{video_stem}",
                    "video_path": str(video_path),
                    "duration": row.get("duration"),
                    "video_type": task_folder,
                    "conversations": [],
                }

            conversation = {
                "question_idx": f"{task_folder}:{video_stem}:{idx}",
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row["question_type"],
            }
            if choices is not None:
                conversation["choices"] = choices
            if "scoring_points" in row:
                conversation["scoring_points"] = row["scoring_points"]
            samples[key]["conversations"].append(conversation)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(list(samples.values()), f, ensure_ascii=False, indent=2)

    missing_path = output.with_name(output.stem + "_missing_videos.txt")
    with missing_path.open("w") as f:
        f.write("\n".join(missing_videos))
        if missing_videos:
            f.write("\n")

    total_questions = sum(len(sample["conversations"]) for sample in samples.values())
    print(f"Wrote {output}")
    print(f"Videos: {len(samples)}")
    print(f"Questions: {total_questions}")
    print(f"Skipped open-ended questions: {skipped_open_ended}")
    print(f"Missing videos: {len(missing_videos)}")
    if missing_videos:
        print(f"Wrote missing list: {missing_path}")


if __name__ == "__main__":
    main()
