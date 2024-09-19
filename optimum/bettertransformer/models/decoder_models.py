# Copyright 2023 The HuggingFace and Meta Team.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotAttention
from transformers.models.bloom.modeling_bloom import BloomAttention
from transformers.models.codegen.modeling_codegen import CodeGenAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention
from transformers.models.marian.modeling_marian import MarianAttention
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.pegasus.modeling_pegasus import PegasusAttention
from transformers.models.t5.modeling_t5 import T5Attention

from ...utils.import_utils import check_if_transformers_greater


if check_if_transformers_greater("4.31"):
    from transformers.models.bark.modeling_bark import BarkSelfAttention
else:
    from ...utils.dummy_bettertransformer_objects import BarkSelfAttention

from .attention import (
    bark_wrapped_scaled_dot_product,
    bart_forward,
    bloom_forward,
    codegen_wrapped_scaled_dot_product,
    gpt2_wrapped_scaled_dot_product,
    gpt_neo_wrapped_scaled_dot_product,
    opt_forward,
    t5_forward,
)
from .base import BetterTransformerBaseLayer


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class GPT2AttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPT2Attention):
    _attn = gpt2_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.is_cross_attention, layer.layer_idx)

        submodules = ["c_proj", "c_attn", "attn_dropout", "resid_dropout", "bias", "masked_bias"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if layer.is_cross_attention:
            setattr(self, "q_attn", getattr(layer, "q_attn"))
            self.original_layers_mapping["q_attn"] = "q_attn"

        self.downcast_qk = False
        self.dropout_prob_attn = config.attn_pdrop

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class GPTJAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTJAttention, nn.Module):
    _attn = gpt2_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        submodules = [
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "attn_dropout",
            "resid_dropout",
            "bias",
            "scale_attn",
            "masked_bias",
        ]
        # Attribute only for transformers>=4.28
        if hasattr(layer, "embed_positions"):
            submodules.append("embed_positions")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.downcast_qk = True
        self.dropout_prob_attn = config.attn_pdrop

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class GPTNeoXAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTNeoXAttention, nn.Module):
    _attn = gpt2_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.module_mapping = None
        submodules = ["rotary_emb", "query_key_value", "dense", "bias", "masked_bias", "norm_factor"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.downcast_qk = True
        self.dropout_prob_attn = 0.0  # no dropout for gpt-neox

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class GPTNeoAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTNeoSelfAttention, nn.Module):
    _attn = gpt_neo_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        if layer.bias[0][0][-1][0] == 1:
            self.attention_type = "global"
        else:
            self.attention_type = "local"

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, self.attention_type)

        self.module_mapping = None
        submodules = ["attn_dropout", "resid_dropout", "k_proj", "v_proj", "q_proj", "out_proj", "bias", "masked_bias"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.scale = torch.sqrt(torch.tensor(layer.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.dropout_prob_attn = float(config.attention_dropout)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class BarkAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BarkSelfAttention, nn.Module):
    _attn = bark_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig", is_causal: bool = False):
        super().__init__(config)

        is_causal = layer.is_causal

        config.dropout = layer.dropout

        config.hidden_size = layer.embed_dim
        config.num_heads = layer.num_heads
        config.bias = layer.out_proj.bias is not None

        if is_causal:
            config.block_size = layer.bias.shape[-1]

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, is_causal)

        self.module_mapping = None
        submodules = ["dropout", "attn_dropout", "resid_dropout", "att_proj", "out_proj"]

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if is_causal:
            setattr(self, "bias", getattr(layer, "bias"))
            self.original_layers_mapping["bias"] = "bias"

        self.supports_training = False
        self.dropout_prob_attn = float(config.dropout)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class BloomAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BloomAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.dropout_prob_attn = config.attention_dropout

        self.module_mapping = None
        self.layer_idx = getattr(layer, "layer_idx", None)

        submodules = ["query_key_value", "dense", "attention_dropout"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

    def forward(self, *args, **kwargs):
        return bloom_forward(self, *args, **kwargs)


class CodegenAttentionLayerBetterTransformer(BetterTransformerBaseLayer, CodeGenAttention, nn.Module):
    _attn = codegen_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.module_mapping = None
        submodules = ["attn_dropout", "resid_dropout", "qkv_proj", "out_proj", "causal_mask", "scale_attn"]

        # Attribute only for transformers>=4.28
        if hasattr(layer, "embed_positions"):
            submodules.append("embed_positions")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.dropout_prob_attn = config.attn_pdrop

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

# Hack
class BLIP2OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        # Access nested attributes from text_config if available
        if hasattr(config, "text_config"):
            text_config = config.text_config
            self.embed_dim = text_config.hidden_size
            self.num_heads = text_config.num_attention_heads
        else:
            self.embed_dim = config.hidden_size
            self.num_heads = config.num_attention_heads

        # Fallback to default values if not found in config
        self.dropout = getattr(config, "attention_dropout", 0.0)
        self.enable_bias = getattr(config, "enable_bias", True)

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states = None,
        past_key_value = None,
        attention_mask = None,
        layer_head_mask = None,
        output_attentions = False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

# This class can only be used with BLIP2 now
class OPTAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BLIP2OPTAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(
                config,
                layer.is_decoder,
            )

        self.scale = torch.sqrt(torch.tensor(layer.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

    def forward(self, *args, **kwargs):
        return opt_forward(self, *args, **kwargs)


class T5AttentionLayerBetterTransformer(BetterTransformerBaseLayer, T5Attention, torch.nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        if hasattr(config, "text_config"):
            config = config.text_config
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.has_relative_attention_bias)

        submodules = ["q", "k", "v", "o"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        head_dim = layer.d_model // layer.n_heads  # hidden size / num attention heads
        self.scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if layer.has_relative_attention_bias:
            setattr(self, "relative_attention_bias", layer.relative_attention_bias)
            self.original_layers_mapping["relative_attention_bias"] = "relative_attention_bias"

        self.module_mapping = None

        self.is_decoder = layer.is_decoder

    def forward(self, *args, **kwargs):
        return t5_forward(self, *args, **kwargs)


def bart_bettertransformer_init(self, layer: "nn.Module", config: "PretrainedConfig"):
    with torch.device("meta"):
        super(BetterTransformerBaseLayer, self).__init__(
            layer.embed_dim,
            layer.num_heads,
            layer.dropout,
            layer.is_decoder,
            layer.k_proj.bias is not None,
        )

    self.module_mapping = None
    submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
    for attr in submodules:
        setattr(self, attr, getattr(layer, attr))

    self.original_layers_mapping = {submodule: submodule for submodule in submodules}

    self.is_decoder = layer.is_decoder


class BartAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BartAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


class BlenderbotAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BlenderbotAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


class M2M100AttentionLayerBetterTransformer(BetterTransformerBaseLayer, M2M100Attention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


class MarianAttentionLayerBetterTransformer(BetterTransformerBaseLayer, MarianAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


class PegasusAttentionLayerBetterTransformer(BetterTransformerBaseLayer, PegasusAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)
