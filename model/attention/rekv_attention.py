import copy
import torch
from typing import Optional

from .kv_cache_manager import ContextManager, ThreeBranchCache
from .dot_production_attention import get_multi_stage_dot_production_attention


def rekv_attention_forward(
    n_local, n_init, topk, chunk_size,
    block_size, max_cached_block,
    exc_block_size, fattn,
    async_global_stream=True,
    pin_memory=False,
    *args, **kwargs
):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv,
    ):

        """ 1. Project QKV """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
            position_bias = copy.deepcopy(position_bias)
            if position_bias.inv_freq.device != h_q.device:
                position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
            if position_bias._cos_cached is not None:
                position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
            if position_bias._sin_cached is not None:
                position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias,
                n_init, n_local, 
                block_size, max_cached_block, topk, chunk_size, exc_block_size,
                fattn,
                async_global_stream,
                pin_memory,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        def finalize_attention_output(score):
            score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            score = score.reshape(batch_size, len_q, num_heads * dim_head)
            return attention_out(score)

        def apply_branch_rope(q_tensor, k_tensor):
            total_len = k_tensor.size(-2) + q_tensor.size(-2)
            cos, sin = position_bias._update_cos_sin_tables_len(total_len, q_tensor.device, q_tensor.dim())
            branch_q = position_bias.apply_rotary_pos_emb(q_tensor, q_tensor.size(-2), total_len, cos, sin)
            branch_k = position_bias.apply_rotary_pos_emb(k_tensor, k_tensor.size(-2), k_tensor.size(-2), cos, sin)
            return branch_q, branch_k

        def run_three_branch_attention(local_h_q, local_h_k, local_h_v, branch_cache):
            attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            stages = [
                {
                    "q": local_h_q,
                    "k": local_h_k,
                    "v": local_h_v,
                    "sliding_window": n_local,
                }
            ]

            if branch_cache.retrieved_k.size(-2) > 0:
                retrieved_h_q, retrieved_h_k = apply_branch_rope(h_q, branch_cache.retrieved_k)
                stages.append(
                    {
                        "q": retrieved_h_q,
                        "k": retrieved_h_k,
                        "v": branch_cache.retrieved_v,
                        "sliding_window": None,
                    }
                )

            if branch_cache.init_k.size(-2) > 0:
                init_h_q = position_bias.apply_rotary_pos_emb_one_angle(h_q, n_local)
                stages.append(
                    {
                        "q": init_h_q,
                        "k": branch_cache.init_k,
                        "v": branch_cache.init_v,
                        "sliding_window": None,
                    }
                )

            for idx, stage in enumerate(stages):
                attn.append(
                    stage["q"],
                    stage["k"],
                    stage["v"],
                    sliding_window=stage["sliding_window"],
                    end=idx == len(stages) - 1,
                )

            score, _ = attn.get_result()
            return finalize_attention_output(score)

        # NOTE: Question-answering / generation with retrieved memory
        if isinstance(past_key_value, ContextManager) and past_key_value.to_retrieve:
            if past_key_value.retrieved_block_indices is None:  # retrieve based on question query
                branch_cache = past_key_value.get_retrieved_kv(global_q)
            else:  # retrieve based on pre-computed block indices
                branch_cache = past_key_value.get_retrieved_kv()

            local_h_q, local_h_k = position_bias(h_q, h_k)
            local_h_v = h_v
            o = run_three_branch_attention(local_h_q, local_h_k, local_h_v, branch_cache)
            return o, branch_cache

        if isinstance(past_key_value, ThreeBranchCache):
            if past_key_value.local_k is not None:
                local_cache_k = torch.cat([past_key_value.local_k, h_k], dim=-2)
                local_cache_v = torch.cat([past_key_value.local_v, h_v], dim=-2)
            else:
                local_cache_k = h_k
                local_cache_v = h_v

            current_key_value = past_key_value.with_local_cache(local_cache_k, local_cache_v, n_local)
            local_h_q, local_h_k = position_bias(h_q, local_cache_k)
            local_h_v = local_cache_v
            o = run_three_branch_attention(local_h_q, local_h_k, local_h_v, current_key_value)
            return o, current_key_value

        # NOTE: Question-answering without retrieved-memory branching, fall back to sliding-window attention (infinite_lm)
        if not isinstance(past_key_value, ContextManager):
            past_k = past_key_value[0]
            past_v = past_key_value[1]

            """ 2. Update KV w/ past KV cache """
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            len_k += past_k.shape[2]

            """ 3. Update KV cache """
            if len_k <= n_local + n_init:
                h_k_cache = h_k
                h_v_cache = h_v
            else:
                h_k_cache = torch.cat([h_k[:, :, :n_init, :], h_k[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                h_v_cache = torch.cat([h_v[:, :, :n_init, :], h_v[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
            current_key_value = (h_k_cache, h_v_cache)

            """ 4. Get local QKV and apply RoPE to local QK """
            h_q_, h_k_, h_v_ = h_q, h_k, h_v
            if len_q + n_local < h_k_.size(-2):
                h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local:, :]
                h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local:, :]

            local_h_q, local_h_k = position_bias(h_q_, h_k_)
            local_h_v = h_v_

            """ 5. Get init QKV and apply RoPE to init Q (Infinite-LM assigns the same position_ids to initial tokens) """
            if len_k > n_local:
                init_h_q = position_bias.apply_rotary_pos_emb_one_angle(
                    h_q, n_local
                )
                init_h_k = h_k[:, :, :n_init, :].contiguous()
                init_h_v = h_v[:, :, :n_init, :].contiguous()
            else:
                init_h_q = h_q
                init_h_k = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_k.device,
                    dtype=h_k.dtype
                )
                init_h_v = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_v.device,
                    dtype=h_v.dtype
                )

            """ 6. Sliding Window Attention """
            attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local)
            attn.append(init_h_q, init_h_k, init_h_v, end=True, sliding_window=(len_k - len_q, n_local), complement_sliding_window=True)
            score, _ = attn.get_result()

            o = finalize_attention_output(score)
            return o, current_key_value

        # NOTE: Encode video, managed by the KVCacheManager
        else:
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )
            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            o = o.reshape(batch_size, len_q, dim_head * num_heads)
            o = attention_out(o)

            return o, past_key_value

    return forward
