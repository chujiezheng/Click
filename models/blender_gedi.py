# coding=utf-8

import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from models.blender_dexperts import Model as DExperts
from transformers.models.blenderbot.modeling_blenderbot import (BlenderbotConfig, BlenderbotForConditionalGeneration,)

import inspect
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers.generation.beam_constraints import Constraint
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.utils import ModelOutput, logging
from transformers.generation.utils import SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput


logger = logging.get_logger(__name__)

class Model(DExperts):
    def __init__(self, config: BlenderbotConfig, expert_path, antiexpert_path, alpha):
        super().__init__(config, expert_path, antiexpert_path, alpha)

    def sample(
        self,
        input_ids: torch.LongTensor,
        expert_input_ids: torch.LongTensor,
        antiexpert_input_ids: torch.LongTensor,
        num_return_sequences: int = 1,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        model_kwargs: dict = None,
        expert_kwargs: dict = None,
        antiexpert_kwargs: dict = None,
    ) -> Union[SampleOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        assert expert_input_ids.shape[0] == antiexpert_input_ids.shape[0]
        assert antiexpert_input_ids.shape[0] % input_ids.shape[0] == 0
        batch_size = input_ids.shape[0]
        multiple = antiexpert_input_ids.shape[0] // input_ids.shape[0]

        expert_scores = input_ids.new_ones((batch_size, ), dtype=torch.float)
        antiexpert_scores = input_ids.new_ones((batch_size, ), dtype=torch.float)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs)
            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(antiexpert_input_ids, **antiexpert_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            expert_outputs = self.expert(
                **expert_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            antiexpert_outputs = self.antiexpert(
                **antiexpert_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]
            expert_next_token_logits = expert_outputs.logits[:, -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[:, -1, :]

            if multiple > 1:
                raise ValueError
                # reshape and average
                aux_attention_mask = antiexpert_inputs['attention_mask'].view(batch_size // num_return_sequences, multiple, num_return_sequences, -1)
                aux_attention_mask = (aux_attention_mask.sum(dim=-1, keepdims=True) > 0).type_as(aux_attention_mask)

                expert_next_token_logits = expert_next_token_logits.view(batch_size // num_return_sequences, multiple, num_return_sequences, -1)
                expert_next_token_logits = (expert_next_token_logits * aux_attention_mask).sum(dim=1) / aux_attention_mask.sum(dim=1)
                expert_next_token_logits = expert_next_token_logits.view(batch_size, -1)

                antiexpert_next_token_logits = antiexpert_next_token_logits.view(batch_size // num_return_sequences, multiple, num_return_sequences, -1)
                antiexpert_next_token_logits = (antiexpert_next_token_logits * aux_attention_mask).sum(dim=1) / aux_attention_mask.sum(dim=1)
                antiexpert_next_token_logits = antiexpert_next_token_logits.view(batch_size, -1)

            # apply modification
            expert_next_token_prob = expert_scores.unsqueeze(-1) * torch.softmax(expert_next_token_logits, dim=-1) / cur_len
            antiexpert_next_token_prob = antiexpert_scores.unsqueeze(-1) * torch.softmax(antiexpert_next_token_logits, dim=-1) / cur_len
            next_token_logits = next_token_logits + self.alpha * (
                    expert_next_token_logits - torch.clamp(expert_next_token_prob + antiexpert_next_token_prob, min=1e-5).log())

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            expert_input_ids = input_ids.unsqueeze(1).repeat(1, multiple, 1).view(-1, input_ids.shape[1])
            antiexpert_input_ids = input_ids.unsqueeze(1).repeat(1, multiple, 1).view(-1, input_ids.shape[1])
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            expert_kwargs = self.expert._update_model_kwargs_for_generation(
                expert_outputs, expert_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            antiexpert_kwargs = self.antiexpert._update_model_kwargs_for_generation(
                antiexpert_outputs, antiexpert_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            expert_scores = expert_scores * expert_next_token_prob[range(next_tokens.size(0)), next_tokens]
            antiexpert_scores = antiexpert_scores * antiexpert_next_token_prob[range(next_tokens.size(0)), next_tokens]
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
