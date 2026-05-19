import warnings
import random
import json
import os
import math
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    logging,
    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
    VideoLlavaForConditionalGeneration, VideoLlavaProcessor
)
import logzero
from logzero import logger

from model import llava_onevision_rekv, video_llava_rekv, longva_rekv, qwen2_5_vl_rekv


MODELS = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-7b-ov-hf',
    },
    'llava_ov_72b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
    },
    'video_llava_7b': {
        'load_func': video_llava_rekv.load_model,
        'model_class': VideoLlavaForConditionalGeneration,
        'processor_class': VideoLlavaProcessor,
        'model_path': 'model_zoo/Video-LLaVA-7B-hf',
    },
    'longva_7b': {
        'load_func': longva_rekv.load_model,
        'model_path': 'model_zoo/LongVA-7B',
    },
    'qwen2_5_vl_7b': {
        'load_func': qwen2_5_vl_rekv.load_model,
        'model_path': '/mnt/models/qwen/Qwen2.5-VL-7B-Instruct',
    },
}


class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1,
                 resize_frame_size=None,
                 save_choice_scores=False) -> None:
        
        self.sample_fps = sample_fps

        self.qa_model = qa_model
        self.qa_processor = qa_processor

        # Retrieval Hyperparams
        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size
        self.resize_frame_size = resize_frame_size
        self.save_choice_scores = save_choice_scores
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        anno = self.normalize_anno_schema(anno)
        self.anno = anno
        self.eval_grounding = (
            bool(anno)
            and 'conversations' in anno[0]
            and len(anno[0]['conversations']) > 0
            and 'temporal_windows' in anno[0]['conversations'][0]
        )

        self.save_dir = save_dir
        self.output_path = f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv'
        self.csv_fieldnames = [
            'video_id',
            'question',
            'choices',
            'answer',
            'correct_choice',
            'pred_answer',
            'pred_choice',
            'qa_acc',
            'task',
            'retrieve_size',
            'chunk_size',
        ]
        if self.save_choice_scores:
            self.csv_fieldnames.extend([
                'choice_logits',
                'choice_logprobs',
                'choice_probs',
                'top1_prob',
                'top2_prob',
                'prob_margin',
                'logit_margin',
                'choice_entropy',
                'normalized_choice_entropy',
            ])

    def normalize_anno_schema(self, anno):
        if not anno:
            return anno
        first = anno[0]
        if 'conversations' in first:
            return anno
        if 'qa' in first and 'downloaded_video_path' in first:
            return [self.convert_lvbench_sample(sample) for sample in anno]
        return anno

    def convert_lvbench_sample(self, sample):
        conversations = []
        for qa in sample.get('qa', []):
            question_text, choices = self.parse_multiple_choice_question(qa['question'])
            conv = {
                'question': question_text,
                'answer': qa.get('answer'),
                'question_type': ', '.join(qa.get('question_type', [])),
            }
            if choices:
                conv['choices'] = choices
            conversations.append(conv)
        return {
            'video_id': sample.get('key'),
            'video_path': sample.get('downloaded_video_path'),
            'conversations': conversations,
        }

    def parse_multiple_choice_question(self, question):
        lines = [line.strip() for line in question.splitlines() if line.strip()]
        if len(lines) <= 1:
            return question.strip(), []

        choice_prefixes = tuple(f"({letter})" for letter in self.choice_letters)
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

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def resize_video_to_square(self, video, frame_size):
        if video.shape[1] == frame_size and video.shape[2] == frame_size:
            return video

        resized = np.empty((video.shape[0], frame_size, frame_size, video.shape[3]), dtype=video.dtype)
        batch_size = 256
        for st in range(0, video.shape[0], batch_size):
            ed = min(st + batch_size, video.shape[0])
            frames = torch.from_numpy(video[st:ed]).permute(0, 3, 1, 2).float()
            frames = F.interpolate(frames, size=(frame_size, frame_size), mode='bilinear', align_corners=False)
            frames = frames.round().clamp(0, 255).to(torch.uint8)
            resized[st:ed] = frames.permute(0, 2, 3, 1).numpy()

        return resized

    def load_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
        video = vr.get_batch(frame_idx).asnumpy()
        if self.resize_frame_size is not None:
            video = self.resize_video_to_square(video, self.resize_frame_size)
        logger.debug(f'video shape: {video.shape}')
        return video
    
    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
        total_intersection_length = 0.0
    
        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = (s or '').strip()
        if not s:
            logger.warning("Empty multiple-choice prediction encountered.")
            return ''
        s_upper = s.upper()

        for idx, ch in enumerate(s_upper):
            if ch == ")" and idx > 0:
                pred = s_upper[idx - 1]
                if pred in self.choice_letters:
                    return pred

        for ch in s_upper:
            if ch in self.choice_letters:
                return ch

        logger.warning("Unable to parse multiple-choice prediction: %r", s)
        return ''

    def video_open_qa(self, question, max_new_tokens=1024):
        pass

    def video_close_qa(self, question, candidates, correct_choice):
        pass

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)

    def append_result(self, row):
        normalized_row = {field: '' for field in self.csv_fieldnames}
        normalized_row.update(row)
        normalized_row['retrieve_size'] = self.retrieve_size
        normalized_row['chunk_size'] = self.chunk_size
        for key, value in list(normalized_row.items()):
            if isinstance(value, (list, dict)):
                normalized_row[key] = json.dumps(value, ensure_ascii=False)

        file_exists = os.path.exists(self.output_path)
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(normalized_row)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def slice_anno_from_video_id(anno, start_video_id):
    for idx, sample in enumerate(anno):
        sample_id = sample.get('video_id') or sample.get('key')
        if sample_id == start_video_id:
            return anno[idx:]
    raise ValueError(f"start_video_id not found in annotation: {start_video_id}")

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--local_block_count", type=int, default=None)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--internal_block_size", type=int, default=0)
    parser.add_argument("--start_video_id", type=str, default=None)
    parser.add_argument("--save_choice_scores", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    args = parser.parse_args()
    args.model = args.model.strip()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    load_kwargs = dict(
        model_path=model_path,
        n_local=args.n_local,
        topk=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
    )
    if args.model == "qwen2_5_vl_7b":
        load_kwargs["local_block_count"] = args.local_block_count
        load_kwargs["frame_size"] = args.frame_size
        load_kwargs["internal_block_size"] = args.internal_block_size or None
    videoqa_model, videoqa_processor = load_func(**load_kwargs)

    # Load ground truth file
    anno = json.load(open(args.anno_path))
    if args.start_video_id:
        original_len = len(anno)
        anno = slice_anno_from_video_id(anno, args.start_video_id)
        logger.info(
            "Restarting from video_id=%s (%d -> %d samples)",
            args.start_video_id,
            original_len,
            len(anno),
        )

    retrieve_analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        retrieve_size=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
        resize_frame_size=args.frame_size if args.model == "qwen2_5_vl_7b" else None,
        save_choice_scores=args.save_choice_scores,
    )

    retrieve_analyzer.analyze(debug=args.debug)
