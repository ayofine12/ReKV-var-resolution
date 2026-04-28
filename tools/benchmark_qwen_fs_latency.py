#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from decord import VideoReader, cpu

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.qwen2_5_vl_rekv import load_model


CHOICE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


def parse_multiple_choice_question(question):
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    if len(lines) <= 1:
        return question.strip(), []

    choice_prefixes = tuple(f"({letter})" for letter in CHOICE_LETTERS)
    choice_start = None
    for idx, line in enumerate(lines):
        if line.startswith(choice_prefixes):
            choice_start = idx
            break

    if choice_start is None:
        return question.strip(), []

    question_text = ' '.join(lines[:choice_start]).strip()
    choices = []
    for line in lines[choice_start:]:
        if line.startswith(choice_prefixes):
            choices.append(line[3:].strip())
        elif choices:
            choices[-1] = f"{choices[-1]} {line}".strip()
    return question_text, choices


def format_mcqa_prompt(question, candidates, qa_model):
    formatted_choices = "\n".join(
        f"({CHOICE_LETTERS[i]}) {candidate}" for i, candidate in enumerate(candidates)
    )
    formatted_question = (
        f"Question: {question}\n"
        f"Options:\n{formatted_choices}\n"
        "Only give the best option."
    )
    return {
        "question": question,
        "formatted_question": formatted_question,
        "prompt": qa_model.get_prompt(formatted_question, mc=True),
    }


def normalize_lvbench_sample(sample):
    conversations = []
    for qa in sample.get('qa', []):
        question_text, choices = parse_multiple_choice_question(qa['question'])
        conv = {
            'question': question_text,
            'choices': choices,
            'answer': qa.get('answer'),
            'task': ', '.join(qa.get('question_type', [])),
        }
        conversations.append(conv)
    return {
        'video_id': sample.get('key'),
        'video_path': sample.get('downloaded_video_path'),
        'conversations': conversations,
    }


def load_sample(anno_path, video_id=None):
    anno = json.load(open(anno_path))
    if video_id is None:
        return normalize_lvbench_sample(anno[0])
    for sample in anno:
        if sample.get('key') == video_id:
            return normalize_lvbench_sample(sample)
    raise ValueError(f"video_id not found in annotation: {video_id}")


def load_video(video_path, sample_fps):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = round(vr.get_avg_fps())
    frame_step = max(1, int(fps / sample_fps))
    frame_idx = [i for i in range(0, len(vr), frame_step)]
    video = vr.get_batch(frame_idx).asnumpy()
    return video, fps, frame_step


def cuda_sync(device):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_gb(device):
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def timed_call(fn, device):
    cuda_sync(device)
    start = time.perf_counter()
    out = fn()
    cuda_sync(device)
    return out, time.perf_counter() - start


def summarize_ms(values):
    ms = [v * 1000.0 for v in values]
    if len(ms) == 1:
        return ms[0], 0.0
    return statistics.mean(ms), statistics.pstdev(ms)


def get_device(model):
    return next(model.parameters()).device


def benchmark_single_frame(model, video_tensor, device, repeats):
    frame_chunk = video_tensor[:1]
    observed_tokens = []
    latencies = []
    peaks = []

    def run_once():
        pixel_values_videos, video_grid_thw = model._prepare_video_inputs(frame_chunk)
        return model._get_video_features(pixel_values_videos, video_grid_thw)

    for _ in range(repeats):
        reset_peak_memory(device)
        video_features, elapsed = timed_call(run_once, device)
        observed_tokens.append(int(video_features.shape[1]))
        latencies.append(elapsed)
        peaks.append(get_peak_memory_gb(device))

    latency_mean_ms, latency_std_ms = summarize_ms(latencies)
    selected_idx = repeats - 1
    return {
        'observed_tokens': observed_tokens[selected_idx],
        'latency_ms_by_repeat': [value * 1000.0 for value in latencies],
        'peak_memory_gb_by_repeat': peaks,
        'selected_repeat': selected_idx + 1,
        'latency_selected_ms': latencies[selected_idx] * 1000.0,
        'peak_memory_selected_gb': peaks[selected_idx],
        'latency_mean_ms': latency_mean_ms,
        'latency_std_ms': latency_std_ms,
        'peak_memory_gb_mean': statistics.mean(peaks),
    }


def benchmark_encode(model, video_tensor, device, repeats):
    init_latencies = []
    encode_latencies = []
    peaks = []

    for _ in range(repeats):
        model.clear_cache()
        reset_peak_memory(device)
        _, init_elapsed = timed_call(model.encode_init_prompt, device)
        _, encode_elapsed = timed_call(lambda: model.encode_video(video_tensor), device)
        init_latencies.append(init_elapsed)
        encode_latencies.append(encode_elapsed)
        peaks.append(get_peak_memory_gb(device))

    init_mean_ms, init_std_ms = summarize_ms(init_latencies)
    encode_mean_ms, encode_std_ms = summarize_ms(encode_latencies)
    num_frames = int(video_tensor.shape[0])
    selected_idx = repeats - 1

    return {
        'selected_repeat': selected_idx + 1,
        'init_ms_by_repeat': [value * 1000.0 for value in init_latencies],
        'encode_ms_by_repeat': [value * 1000.0 for value in encode_latencies],
        'peak_memory_gb_by_repeat': peaks,
        'init_selected_ms': init_latencies[selected_idx] * 1000.0,
        'encode_selected_ms': encode_latencies[selected_idx] * 1000.0,
        'encode_selected_ms_per_frame': (encode_latencies[selected_idx] * 1000.0) / max(1, num_frames),
        'peak_memory_selected_gb': peaks[selected_idx],
        'init_mean_ms': init_mean_ms,
        'init_std_ms': init_std_ms,
        'encode_mean_ms': encode_mean_ms,
        'encode_std_ms': encode_std_ms,
        'encode_ms_per_frame': encode_mean_ms / max(1, num_frames),
        'peak_memory_gb_mean': statistics.mean(peaks),
    }


def benchmark_qa(model, sample, video_tensor, device, question_limit, repeats):
    questions = []
    for conv in sample['conversations']:
        if conv['choices']:
            questions.append(conv)
        if len(questions) >= question_limit:
            break
    if not questions:
        raise ValueError("No multiple-choice questions found for the selected video.")

    qa_latencies = []
    qa_mean_by_repeat = []
    peaks = []

    for _ in range(repeats):
        model.clear_cache()
        model.encode_init_prompt()
        model.encode_video(video_tensor)
        reset_peak_memory(device)
        repeat_latencies = []
        for conv in questions:
            input_text = format_mcqa_prompt(conv['question'], conv['choices'], model)
            _, elapsed = timed_call(lambda: model.question_answering(input_text, max_new_tokens=16), device)
            qa_latencies.append(elapsed)
            repeat_latencies.append(elapsed)
        qa_mean_by_repeat.append(sum(repeat_latencies) / len(repeat_latencies))
        peaks.append(get_peak_memory_gb(device))

    qa_mean_ms, qa_std_ms = summarize_ms(qa_latencies)
    selected_idx = repeats - 1
    return {
        'question_count': len(questions),
        'selected_repeat': selected_idx + 1,
        'qa_mean_ms_by_repeat': [value * 1000.0 for value in qa_mean_by_repeat],
        'peak_memory_gb_by_repeat': peaks,
        'qa_selected_ms': qa_mean_by_repeat[selected_idx] * 1000.0,
        'peak_memory_selected_gb': peaks[selected_idx],
        'qa_mean_ms': qa_mean_ms,
        'qa_std_ms': qa_std_ms,
        'peak_memory_gb_mean': statistics.mean(peaks),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--frame_size", type=int, required=True)
    parser.add_argument("--local_block_count", type=int, required=True)
    parser.add_argument("--retrieve_size", type=int, required=True)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--question_limit", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    sample = load_sample(args.anno_path, video_id=args.video_id)
    print(f"video_id: {sample['video_id']}")
    print(f"video_path: {sample['video_path']}")

    model, _ = load_model(
        model_path='/mnt/models/qwen/Qwen2.5-VL-7B-Instruct',
        local_block_count=args.local_block_count,
        topk=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        frame_size=args.frame_size,
    )
    device = get_device(model)

    video_np, fps, frame_step = load_video(sample['video_path'], args.sample_fps)
    video_tensor = torch.from_numpy(video_np)
    sampled_frames = int(video_tensor.shape[0])

    single_frame = benchmark_single_frame(model, video_tensor, device, args.repeats)
    encode = benchmark_encode(model, video_tensor, device, args.repeats)
    qa = benchmark_qa(model, sample, video_tensor, device, args.question_limit, args.repeats)

    print("")
    print("[config]")
    print(f"frame_size: {args.frame_size}")
    print(f"local_block_count: {args.local_block_count}")
    print(f"retrieve_size: {args.retrieve_size}")
    print(f"retrieve_chunk_size: {args.retrieve_chunk_size}")
    print(f"sample_fps: {args.sample_fps}")
    print(f"repeats: {args.repeats}")

    print("")
    print("[video]")
    print(f"original_fps_rounded: {fps}")
    print(f"sampling_step: every {frame_step} frame(s)")
    print(f"sampled_frames: {sampled_frames}")

    print("")
    print("[tokens]")
    print(f"configured_n_frame_tokens: {model.n_frame_tokens}")
    print(f"observed_single_frame_tokens: {single_frame['observed_tokens']}")
    print(f"encoded_visual_tokens_total: {model.n_frame_tokens * sampled_frames}")
    print(f"retrieved_visual_tokens_budget: {model.n_frame_tokens * args.retrieve_size}")
    print(f"local_visual_tokens_budget: {model.n_frame_tokens * args.local_block_count}")

    print("")
    print("[single_frame_vision]")
    print(f"selected_repeat: {single_frame['selected_repeat']}")
    print(f"latency_ms_selected: {single_frame['latency_selected_ms']:.2f}")
    print(f"peak_memory_gb_selected: {single_frame['peak_memory_selected_gb']:.2f}")
    print(f"latency_ms_mean: {single_frame['latency_mean_ms']:.2f}")
    print(f"latency_ms_std: {single_frame['latency_std_ms']:.2f}")
    print(f"peak_memory_gb_mean: {single_frame['peak_memory_gb_mean']:.2f}")

    print("")
    print("[encode_video]")
    print(f"selected_repeat: {encode['selected_repeat']}")
    print(f"init_prompt_ms_selected: {encode['init_selected_ms']:.2f}")
    print(f"total_ms_selected: {encode['encode_selected_ms']:.2f}")
    print(f"ms_per_sampled_frame_selected: {encode['encode_selected_ms_per_frame']:.2f}")
    print(f"peak_memory_gb_selected: {encode['peak_memory_selected_gb']:.2f}")
    print(f"init_prompt_ms_mean: {encode['init_mean_ms']:.2f}")
    print(f"init_prompt_ms_std: {encode['init_std_ms']:.2f}")
    print(f"total_ms_mean: {encode['encode_mean_ms']:.2f}")
    print(f"total_ms_std: {encode['encode_std_ms']:.2f}")
    print(f"ms_per_sampled_frame: {encode['encode_ms_per_frame']:.2f}")
    print(f"peak_memory_gb_mean: {encode['peak_memory_gb_mean']:.2f}")

    print("")
    print("[question_answering]")
    print(f"question_count: {qa['question_count']}")
    print(f"selected_repeat: {qa['selected_repeat']}")
    print(f"latency_ms_selected: {qa['qa_selected_ms']:.2f}")
    print(f"peak_memory_gb_selected: {qa['peak_memory_selected_gb']:.2f}")
    print(f"latency_ms_mean: {qa['qa_mean_ms']:.2f}")
    print(f"latency_ms_std: {qa['qa_std_ms']:.2f}")
    print(f"peak_memory_gb_mean: {qa['peak_memory_gb_mean']:.2f}")
    print(
        "estimated_end_to_end_ms_for_question_batch_selected: "
        f"{encode['init_selected_ms'] + encode['encode_selected_ms'] + qa['question_count'] * qa['qa_selected_ms']:.2f}"
    )
    print(
        "estimated_end_to_end_ms_for_question_batch_mean: "
        f"{encode['init_mean_ms'] + encode['encode_mean_ms'] + qa['question_count'] * qa['qa_mean_ms']:.2f}"
    )
