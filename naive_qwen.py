#!/usr/bin/env python3
"""Naive Qwen2.5-VL video QA baseline with uniform frame sampling.

This script intentionally does not use ReKV, retrieval, or resolution routing.
It samples a fixed number of frames uniformly from each video, feeds those frames
directly to Qwen2.5-VL, and evaluates multiple-choice answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, logging


CHOICE_LETTERS = string.ascii_uppercase
DEFAULT_MODEL_PATH = "/mnt/models/qwen/Qwen2.5-VL-7B-Instruct"


CSV_FIELDNAMES = [
    "video_id",
    "question",
    "choices",
    "answer",
    "correct_choice",
    "pred_answer",
    "pred_choice",
    "qa_acc",
    "task",
    "num_frames",
    "frame_size",
    "video_path",
    "choice_logits",
    "choice_logprobs",
    "choice_probs",
    "top1_prob",
    "top2_prob",
    "prob_margin",
    "logit_margin",
    "choice_entropy",
    "normalized_choice_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a naive Qwen2.5-VL MCQA video baseline.")
    parser.add_argument("--anno_path", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument(
        "--video_root",
        type=Path,
        default=None,
        help="Optional directory used to resolve relative or missing video paths by basename.",
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument(
        "--frame_size",
        type=int,
        default=224,
        help="Resize sampled frames to a square size before Qwen processing. Use 0 to disable.",
    )
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--start_video_id", default=None)
    parser.add_argument("--limit_videos", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    return parser.parse_args()


def split_list(items: Sequence[Any], n: int) -> List[Sequence[Any]]:
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def select_chunk(items: Sequence[Any], num_chunks: int, chunk_idx: int) -> Sequence[Any]:
    if num_chunks < 1:
        raise ValueError("--num_chunks must be positive.")
    chunks = split_list(items, num_chunks)
    if chunk_idx < 0 or chunk_idx >= len(chunks):
        raise ValueError(f"--chunk_idx must be in [0, {len(chunks) - 1}], got {chunk_idx}.")
    return chunks[chunk_idx]


def slice_from_video_id(anno: Sequence[Dict[str, Any]], start_video_id: str | None) -> List[Dict[str, Any]]:
    if not start_video_id:
        return list(anno)
    for idx, sample in enumerate(anno):
        sample_id = sample.get("video_id") or sample.get("key")
        if sample_id == start_video_id:
            return list(anno[idx:])
    raise ValueError(f"start_video_id not found: {start_video_id}")


def parse_multiple_choice_question(question: str) -> Tuple[str, List[str]]:
    lines = [line.strip() for line in str(question).splitlines() if line.strip()]
    if len(lines) <= 1:
        return str(question).strip(), []
    prefixes = tuple(f"({letter})" for letter in CHOICE_LETTERS)
    choice_start = None
    for idx, line in enumerate(lines):
        if line.startswith(prefixes):
            choice_start = idx
            break
    if choice_start is None:
        return str(question).strip(), []
    question_text = " ".join(lines[:choice_start]).strip()
    choices: List[str] = []
    for line in lines[choice_start:]:
        if line.startswith(prefixes):
            choices.append(line[3:].strip())
        elif choices:
            choices[-1] = f"{choices[-1]} {line}".strip()
    return question_text, choices


def convert_lvbench_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    conversations = []
    for qa in sample.get("qa", []):
        question_text, choices = parse_multiple_choice_question(qa["question"])
        conv: Dict[str, Any] = {
            "question": question_text,
            "answer": qa.get("answer"),
            "question_type": ", ".join(qa.get("question_type", [])),
        }
        if choices:
            conv["choices"] = choices
        conversations.append(conv)
    return {
        "video_id": sample.get("key"),
        "video_path": sample.get("downloaded_video_path"),
        "conversations": conversations,
    }


def normalize_annotation(anno: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not anno:
        return anno
    first = anno[0]
    if "conversations" in first:
        return anno
    if "qa" in first and "downloaded_video_path" in first:
        return [convert_lvbench_sample(sample) for sample in anno]
    raise ValueError("Unsupported annotation schema. Expected conversations or LVBench qa format.")


def load_annotation(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return normalize_annotation(json.load(fh))


def resolve_video_path(video_path: str, video_root: Path | None) -> str:
    path = Path(video_path)
    if path.exists():
        return str(path)
    if video_root is None:
        return str(path)

    candidates = []
    if not path.is_absolute():
        candidates.append(video_root / path)
    candidates.append(video_root / path.name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(path)


def resize_video_to_square(video: np.ndarray, frame_size: int) -> np.ndarray:
    if frame_size <= 0:
        return video
    if video.shape[1] == frame_size and video.shape[2] == frame_size:
        return video
    resized = np.empty((video.shape[0], frame_size, frame_size, video.shape[3]), dtype=video.dtype)
    batch_size = 128
    for start in range(0, video.shape[0], batch_size):
        end = min(start + batch_size, video.shape[0])
        frames = torch.from_numpy(video[start:end]).permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=(frame_size, frame_size), mode="bilinear", align_corners=False)
        frames = frames.round().clamp(0, 255).to(torch.uint8)
        resized[start:end] = frames.permute(0, 2, 3, 1).numpy()
    return resized


def load_uniform_frames(video_path: str, num_frames: int, frame_size: int) -> np.ndarray:
    if num_frames < 1:
        raise ValueError("--num_frames must be positive.")
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    if len(vr) == 0:
        raise ValueError(f"Empty video: {video_path}")
    frame_count = min(num_frames, len(vr))
    frame_idx = np.linspace(0, len(vr) - 1, frame_count, dtype=np.int64).tolist()
    video = vr.get_batch(frame_idx).asnumpy()
    return resize_video_to_square(video, frame_size)


def format_mcqa_prompt(question: str, choices: Sequence[str]) -> str:
    formatted_choices = "\n".join(
        f"({CHOICE_LETTERS[idx]}) {choice}" for idx, choice in enumerate(choices)
    )
    return (
        f"Question: {question}\n"
        f"Options:\n{formatted_choices}\n"
        "Only give the best option."
    )


def build_qwen_text(processor: AutoProcessor, question: str, choices: Sequence[str]) -> str:
    user_text = format_mcqa_prompt(question, choices)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Best option: ("


def choice_token_ids(processor: AutoProcessor, num_choices: int) -> Tuple[str, List[int]]:
    letters = CHOICE_LETTERS[:num_choices]
    token_ids: List[int] = []
    for letter in letters:
        ids = processor.tokenizer(letter, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Expected one token for choice {letter!r}, got {ids}")
        token_ids.append(ids[0])
    return letters, token_ids


def score_choices(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    video: np.ndarray,
    question: str,
    choices: Sequence[str],
) -> Dict[str, Any]:
    text = build_qwen_text(processor, question, choices)
    inputs = processor(text=[text], videos=[video], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(model.dtype)

    output = model(**inputs)
    last_token_logits = output.logits[0, -1, :]
    letters, token_ids = choice_token_ids(processor, len(choices))
    token_tensor = torch.as_tensor(token_ids, device=last_token_logits.device)
    logits = last_token_logits.index_select(0, token_tensor).float()
    logprobs = torch.log_softmax(logits, dim=0)
    probs = logprobs.exp()
    pred_idx = int(torch.argmax(logits).item())
    pred = letters[pred_idx]

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    sorted_logits = logits.index_select(0, sorted_indices)
    top1_prob = float(sorted_probs[0].item())
    top2_prob = float(sorted_probs[1].item()) if len(choices) > 1 else 0.0
    prob_margin = top1_prob - top2_prob
    logit_margin = (
        float((sorted_logits[0] - sorted_logits[1]).item()) if len(choices) > 1 else float("inf")
    )
    entropy = float((-(probs * logprobs).sum()).item())
    normalized_entropy = entropy / math.log(len(choices)) if len(choices) > 1 else 0.0

    return {
        "pred_answer": pred,
        "pred_choice": pred,
        "choice_logits": {letter: float(score.item()) for letter, score in zip(letters, logits)},
        "choice_logprobs": {letter: float(score.item()) for letter, score in zip(letters, logprobs)},
        "choice_probs": {letter: float(score.item()) for letter, score in zip(letters, probs)},
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "prob_margin": prob_margin,
        "logit_margin": logit_margin,
        "choice_entropy": entropy,
        "normalized_choice_entropy": normalized_entropy,
    }


def correct_choice_for(answer: Any, choices: Sequence[str]) -> str:
    answer_text = str(answer).strip()
    valid_letters = CHOICE_LETTERS[: len(choices)]
    if answer_text in valid_letters:
        return answer_text
    if answer_text in choices:
        return CHOICE_LETTERS[choices.index(answer_text)]
    raise ValueError(f"Unable to map answer={answer!r} to one of {len(choices)} choices.")


def output_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("video_id", "")),
        str(row.get("question", "")),
        str(row.get("choices", "")),
        str(row.get("correct_choice", "")),
    )


def load_existing_keys(output_path: Path) -> set[Tuple[str, str, str, str]]:
    if not output_path.exists():
        return set()
    with output_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {output_key(row) for row in reader}


def append_row(output_path: Path, row: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {field: "" for field in CSV_FIELDNAMES}
    normalized.update(row)
    for key, value in list(normalized.items()):
        if isinstance(value, (list, dict)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized)


def iter_rows_for_video(
    sample: Dict[str, Any],
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    args: argparse.Namespace,
) -> Iterable[Dict[str, Any]]:
    video_path = resolve_video_path(sample["video_path"], args.video_root)
    video = load_uniform_frames(video_path, args.num_frames, args.frame_size)
    for conv in sample.get("conversations", []):
        if "choices" not in conv:
            continue
        choices = list(conv["choices"])
        correct_choice = correct_choice_for(conv.get("answer"), choices)
        result = score_choices(model, processor, video, conv["question"], choices)
        yield {
            "video_id": sample["video_id"],
            "question": conv["question"],
            "choices": choices,
            "answer": conv.get("answer"),
            "correct_choice": correct_choice,
            "pred_answer": result["pred_answer"],
            "pred_choice": result["pred_choice"],
            "qa_acc": float(result["pred_choice"] == correct_choice) * 100.0,
            "task": conv.get("question_type", ""),
            "num_frames": args.num_frames,
            "frame_size": args.frame_size,
            "video_path": video_path,
            **{
                key: result[key]
                for key in [
                    "choice_logits",
                    "choice_logprobs",
                    "choice_probs",
                    "top1_prob",
                    "top2_prob",
                    "prob_margin",
                    "logit_margin",
                    "choice_entropy",
                    "normalized_choice_entropy",
                ]
            },
        }


def load_model(args: argparse.Namespace) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(args.model_path)
    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return model, processor


def main() -> None:
    args = parse_args()
    logging.set_verbosity_error()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.save_dir / f"{args.num_chunks}_{args.chunk_idx}.csv"
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    anno = load_annotation(args.anno_path)
    anno = slice_from_video_id(anno, args.start_video_id)
    anno = list(select_chunk(anno, args.num_chunks, args.chunk_idx))
    if args.limit_videos is not None:
        anno = anno[: args.limit_videos]
    if args.debug:
        anno = anno[:1]

    existing_keys = load_existing_keys(output_path) if args.resume else set()
    model, processor = load_model(args)

    total = 0
    correct = 0.0
    skipped = 0
    for sample in tqdm(anno, desc="videos"):
        for row in iter_rows_for_video(sample, model, processor, args):
            key = output_key(
                {
                    **row,
                    "choices": json.dumps(row["choices"], ensure_ascii=False),
                }
            )
            if key in existing_keys:
                skipped += 1
                continue
            append_row(output_path, row)
            total += 1
            correct += float(row["qa_acc"]) / 100.0

    if total:
        print(f"[summary] wrote={total} skipped={skipped} acc={correct / total * 100:.2f}")
    else:
        print(f"[summary] wrote=0 skipped={skipped}")
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
