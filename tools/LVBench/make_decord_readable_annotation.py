#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from decord import VideoReader, cpu


def video_id(item):
    return item.get("video_id") or item.get("key") or item.get("id")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/mnt/ssd1/mwnoh/LVBench/data/video_info.json")
    parser.add_argument("--output", default="/mnt/ssd1/mwnoh/LVBench/data/video_info_decord_ok.json")
    parser.add_argument("--bad-output", default="/mnt/ssd1/mwnoh/LVBench/data/video_info_decord_bad.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    bad_output_path = Path(args.bad_output)

    items = json.loads(input_path.read_text())
    ok_items = []
    bad_items = []

    for index, item in enumerate(items):
        path = Path(item.get("downloaded_video_path", ""))
        vid = video_id(item)
        reason = ""
        if not path.exists():
            reason = "missing"
        else:
            try:
                vr = VideoReader(str(path), ctx=cpu(0))
                _ = len(vr)
            except Exception as exc:
                reason = f"decord_error: {exc}"
        if reason:
            bad_items.append({"index": index, "video_id": vid, "path": str(path), "reason": reason})
        else:
            ok_items.append(item)

    output_path.write_text(json.dumps(ok_items, ensure_ascii=False, indent=2))
    bad_output_path.write_text(json.dumps(bad_items, ensure_ascii=False, indent=2))

    print(f"[summary] input={len(items)} ok={len(ok_items)} bad={len(bad_items)}")
    print(f"[saved_ok] {output_path}")
    print(f"[saved_bad] {bad_output_path}")
    for bad in bad_items:
        print(f"[bad] {bad['index']} {bad['video_id']} {bad['reason']}")


if __name__ == "__main__":
    main()
