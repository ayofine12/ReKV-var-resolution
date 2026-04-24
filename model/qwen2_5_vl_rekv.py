import torch
from logzero import logger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from model.abstract_rekv import Abstract_ReKV
from model.patch import patch_hf


class Qwen2_5_VL_ReKV(Qwen2_5_VLForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor=None, n_frame_tokens=None, init_prompt_ids=None, n_local=None, topk=None, chunk_size=1):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        if processor is not None:
            Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)

    def get_prompt(self, query, mc=False):
        prompt = f"{query}<|im_end|>\n<|im_start|>assistant\n"
        if mc:
            prompt += "Best option: ("
        return prompt

    def _prepare_video_inputs(self, video_chunk):
        video_inputs = self.processor(
            text=["<|vision_start|><|video_pad|><|vision_end|>"],
            videos=[video_chunk],
            padding=True,
            return_tensors="pt",
        )
        pixel_values_videos = video_inputs["pixel_values_videos"].to(self.device, self.dtype)
        video_grid_thw = video_inputs["video_grid_thw"].to(self.device)
        return pixel_values_videos, video_grid_thw

    def _get_video_features(self, pixel_values_videos, video_grid_thw):
        video_features = self.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        ).pooler_output
        video_features = torch.cat(video_features, dim=0).unsqueeze(0)
        return video_features

    def _encode_video_chunk(self, video_chunk):
        pixel_values_videos, video_grid_thw = self._prepare_video_inputs(video_chunk)
        video_features = self._get_video_features(pixel_values_videos, video_grid_thw)
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []

        input_ids = self.processor.tokenizer(input_text["question"]).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        for layer_kv in self.kv_cache:
            layer_kv.set_retrieval()

        if retrieved_indices is None:
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values
        else:
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values

        for layer_kv in self.kv_cache:
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0:
                input_ids = self.processor.tokenizer(input_text["prompt"]).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = self.lm_head(out["last_hidden_state"])
            else:
                out = self.language_model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = self.lm_head(out["last_hidden_state"])
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            _, indices = torch.topk(last_token_logits, 2)
            token = int(indices.tolist()[0])
            output_ids.append(token)
            if token in stop_token_ids:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        return output


def load_model(model_path='/mnt/models/qwen/Qwen2.5-VL-7B-Instruct',
               n_init=None, n_local=None, topk=64, chunk_size=1, frame_size=224):
    device = 'cuda'
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=frame_size * frame_size,
        max_pixels=frame_size * frame_size,
    )
    spatial_unit = processor.video_processor.patch_size * processor.video_processor.merge_size
    assert frame_size % spatial_unit == 0, (
        f"frame_size must be divisible by patch_size * merge_size ({spatial_unit}), got {frame_size}"
    )
    n_frame_tokens = (frame_size // spatial_unit) ** 2

    init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = Qwen2_5_VL_ReKV.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    Abstract_ReKV.__init__(
        model,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )
    model.language_model = patch_hf(model.model.language_model, **inf_llm_config)

    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'frame_size: {frame_size}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()
    return model, processor
