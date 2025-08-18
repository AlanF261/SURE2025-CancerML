# Code primarily copied from Nucleotide Transformer v2 with modifications
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, SiLU
import torch.nn.functional as nnfunc
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput, SequenceClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from config import Config

logger = logging.get_logger(__name__)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def symmetrize(x):
    return x + x.transpose(-1, -2)


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
            self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size, padding_idx=config.pad_token_id
        )

        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(
                self.config.hidden_size, eps=self.config.layer_norm_eps
            )
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.methylation_embeddings = nn.Embedding(
            self.config.max_methylation_embeddings,
            self.config.hidden_size,
            padding_idx=self.config.meth_pad_id,
        )
        self.age_embeddings = nn.Embedding(
            self.config.max_age_embeddings, self.config.hidden_size, padding_idx=self.config.age_pad_id
        )
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask=None,
            methylation_ids=None,
            age_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
    ):

        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
        else:
            embeddings = inputs_embeds

        if methylation_ids is not None:
            methylation_embeddings = self.methylation_embeddings(methylation_ids)
            embeddings += methylation_embeddings

        if age_ids is not None:
            age_embeds = self.age_embeddings(age_ids)
            embeddings += age_embeds

        if self.token_dropout:
            embeddings.masked_fill_(
                (input_ids == self.mask_token_id).unsqueeze(-1), 0.0
            )
            mask_ratio_train = self.config.mask_ratio_train
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(
                -1
            ).float() / src_lengths
            embeddings = (
                    embeddings
                    * (1 - mask_ratio_train)
                    / (1 - mask_ratio_observed)[:, None, None]
            ).to(embeddings.dtype)

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(
                embeddings.dtype
            )

        return embeddings


class AttentionCalculation(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout.p = config.attention_probs_secondary_dropout
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        if self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ):
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = query_layer * self.attention_head_size**-0.5

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        context_layer = nnfunc.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p,
            is_causal=False
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = context_layer

        return (outputs,)


class AttentionResidualOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states


class AttentionMain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = AttentionCalculation(config)
        self.output = AttentionResidualOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
                self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class FeedForwardGating(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            int(config.intermediate_size * 2),
            bias=config.add_bias_fnn,
        )
        self.activation_fn = SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)

        x1, x2 = hidden_states.split(int(hidden_states.size(-1) / 2), -1)
        hidden_states = self.activation_fn(x1) * x2

        return hidden_states


class FeedForwardOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.add_bias_fnn
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AttentionMain(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = AttentionMain(config)
        self.intermediate = FeedForwardGating(config)
        self.output = FeedForwardOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_attn_past_key_value = (
            past_key_value if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [Layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.emb_layer_norm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
        )


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BasePreTrainedModel(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

    config_class = Config
    base_model_prefix = "bc_predictor"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Layer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Model(BasePreTrainedModel):

    supports_gradient_checkpointing = False

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

        self.pooler = Pooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            methylation_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            age_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            methylation_ids=methylation_ids,
            age_ids=age_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)
            if not return_dict:
                return (pooled_output,) + encoder_outputs[1:]

            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
            )

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class HierarchicalGenomeTransformer(BasePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.lower_layer_transformer = Model(config, add_pooling_layer=True)

        self.higher_layer_transformer = Model(config, add_pooling_layer=True)

        self.segment_length = config.segment_length
        self.segment_stride = config.segment_stride

        self.post_init()

    def get_input_embeddings(self):
        return self.lower_layer_transformer.get_input_embeddings()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            methylation_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            age_ids: Optional[torch.Tensor] = None,
    ):
        batch_size, sequence_length = input_ids.shape
        segment_length = self.segment_length
        segment_stride = self.segment_stride

        padding_length = 0
        if (sequence_length - segment_length) % segment_stride != 0:
            padding_length = segment_stride - ((sequence_length - segment_length) % segment_stride)

        padded_input_ids = nn.functional.pad(input_ids, (0, padding_length), 'constant', self.config.pad_token_id)
        padded_attention_mask = nn.functional.pad(attention_mask, (0, padding_length), 'constant', 0)
        padded_methylation_ids = nn.functional.pad(methylation_ids, (0, padding_length), 'constant', self.config.meth_pad_id)

        segmented_input_ids = padded_input_ids.unfold(dimension=1, size=segment_length, step=segment_stride)
        segmented_attention_mask = padded_attention_mask.unfold(dimension=1, size=segment_length, step=segment_stride)
        segmented_methylation_ids = padded_methylation_ids.unfold(dimension=1, size=segment_length, step=segment_stride)

        flat_input_ids = segmented_input_ids.reshape(-1, segment_length)
        flat_attention_mask = segmented_attention_mask.reshape(-1, segment_length)
        flat_methylation_ids = segmented_methylation_ids.reshape(-1, segment_length)

        lower_layer_output = self.lower_layer_transformer(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            methylation_ids=flat_methylation_ids,
        )

        num_segments = segmented_input_ids.shape[1]
        segment_embeddings = lower_layer_output.pooler_output.view(batch_size, num_segments, -1)

        higher_layer_attention_mask = torch.ones(
            batch_size, num_segments, dtype=torch.long, device=input_ids.device
        )

        higher_layer_output = self.higher_layer_transformer(
            inputs_embeds=segment_embeddings,
            attention_mask=higher_layer_attention_mask,
            return_dict=True,
            age_ids=age_ids[:, 0].unsqueeze(-1).expand(-1, num_segments) if age_ids is not None else None,
        )

        return higher_layer_output

class LMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x) + self.bias
        return x


class MaskedLM(BasePreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `MaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.model = Model(config, add_pooling_layer=False)
        self.lm_head = LMHead(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            methylation_ids: Optional[torch.Tensor] = None,
            age_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            methylation_ids=methylation_ids,
            age_ids=age_ids,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CancerClassificationModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MaskedLM(config)
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        self.loss_fct = CrossEntropyLoss()
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            methylation_ids: Optional[torch.Tensor] = None,
            age_ids: Optional[torch.Tensor] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            methylation_ids=methylation_ids,
            age_ids=age_ids,
        )
        sequence_output = outputs.hidden_states[-1]
        pooled_output = sequence_output[:, 0, :]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )