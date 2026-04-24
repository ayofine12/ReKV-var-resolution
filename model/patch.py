import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from model.attention import RotaryEmbeddingESM, rekv_attention_forward


def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions
        ret = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_value,
            self.q_proj, self.k_proj, self.v_proj, self.o_proj, 
            self.head_dim, self.num_heads, self.num_key_value_heads
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None

        return o, None, pkv

    return hf_forward


def patch_hf(
    model,
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen2Model
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, BaseModelOutputWithPast

    def get_decoder_root(model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        if hasattr(model, "layers") and hasattr(model, "embed_tokens"):
            return model
        return None

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        *args,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        if use_cache:
            pkv = tuple()

        else:
            pkv = None

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                _cache = layer_outputs[2 if output_attentions else 1]
                pkv = pkv + (_cache,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def qwen2_5_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    forward = huggingface_forward(rekv_attention_forward(**attn_kwargs))

    decoder_root = None
    DecoderLayer = None
    if isinstance(model, LlamaForCausalLM):
        decoder_root = model.model
    elif isinstance(model, MistralForCausalLM):
        decoder_root = model.model
    elif isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen2Model):
        decoder_root = get_decoder_root(model)
    elif model.__class__.__name__ == "Qwen2_5_VLTextModel":
        decoder_root = get_decoder_root(model)
        DecoderLayer = decoder_root.layers[0].__class__
    elif model.__class__.__name__ == "MiniCPMForCausalLM":
        decoder_root = model.model
    else:
        raise ValueError(f"Only supports llama, mistral and qwen2 models, not {model.__class__.__name__}.")

    Attention = decoder_root.layers[0].self_attn.__class__
    Model = decoder_root.__class__

    hf_rope = getattr(decoder_root.layers[0].self_attn, "rotary_emb", None)
    if hf_rope is None:
        hf_rope = getattr(decoder_root, "rotary_emb", None)
    rope_config = getattr(hf_rope, "config", None)
    if rope_config is None:
        rope_config = getattr(decoder_root, "config", None)

    if isinstance(hf_rope, Qwen2RotaryEmbedding):
        base = hf_rope.base
        distance_scale = 1.0
        dim = hf_rope.dim
    else:
        base = getattr(rope_config, "rope_theta", None)
        if base is None:
            rope_parameters = getattr(rope_config, "rope_parameters", None)
            if rope_parameters is None:
                rope_parameters = getattr(getattr(model, "config", None), "rope_parameters", None)
            if rope_parameters is not None:
                base = rope_parameters.get("rope_theta")
        if base is None:
            base = 10000.0
        distance_scale = distance_scale if distance_scale is not None else 1.0
        partial_rotary_factor = getattr(rope_config, "partial_rotary_factor", 1.0)
        hidden_size = getattr(rope_config, "hidden_size", None)
        num_attention_heads = getattr(rope_config, "num_attention_heads", None)
        if hidden_size is None or num_attention_heads is None:
            hidden_size = model.config.hidden_size
            num_attention_heads = model.config.num_attention_heads
        dim = int((hidden_size // num_attention_heads) * partial_rotary_factor)
    rope = RotaryEmbeddingESM(
        dim,
        base,
        distance_scale
    )
    decoder_root.position_bias = rope

    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)
        if DecoderLayer is not None and isinstance(m, DecoderLayer):
            m._old_forward = m.forward
            m.forward = qwen2_5_decoder_forward.__get__(m, DecoderLayer)

    model.apply(set_forward)

    decoder_root._old_forward = decoder_root.forward
    decoder_root.forward = model_forward.__get__(decoder_root, Model)

    return model
