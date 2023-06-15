# coding=utf-8

import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from models.blender import Model as Expert
from models.blender import Model as AntiExpert
from transformers.models.blenderbot.modeling_blenderbot import (BlenderbotConfig, BlenderbotForConditionalGeneration,)
from utils.model_utils import BaseModel

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

class Model(BaseModel, BlenderbotForConditionalGeneration):
    def __init__(self, config: BlenderbotConfig, expert_path, antiexpert_path, alpha):
        super().__init__(config)
        self.expert_path = expert_path
        self.antiexpert_path = antiexpert_path
        self.alpha = float(alpha)

    def tie_tokenizer_and_post_init(self, toker, process_index=0):
        super().tie_tokenizer_and_post_init(toker, process_index)
        self.expert.tie_tokenizer_and_post_init(toker, process_index)
        self.antiexpert.tie_tokenizer_and_post_init(toker, process_index)

    def init_new_layers(self):
        self.expert = Expert.from_pretrained(self.expert_path)
        self.antiexpert = AntiExpert.from_pretrained(self.antiexpert_path)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=True,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        assert not (self.training and validation)
        if self.training or validation:
            assert labels is not None

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            labels=None,
            return_dict=return_dict,
            use_cache=use_cache,
            **kwargs
        )

        lm_logits = outputs.logits
        masked_lm_loss = None
        if self.training or validation:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = labels.ne(-100).sum(1).type_as(loss)
            masked_lm_loss = loss.sum() / torch.clamp(label_size.sum(), min=1e-5)
            ppl_value = masked_lm_loss.exp()

        outputs.loss = masked_lm_loss

        if not self.training and not validation: # inference
            return outputs
        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res
        else: # validation
            assert not self.training
            return loss, label_size

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        **kwargs
    ):
        """
        (input_ids, attention_mask, decoder_input_ids)
        (input_ids, attention_mask)
        """
        assert not self.training
        assert self.toker is not None
        assert input_ids.size(0) == 1 or ((input_ids is None) == (attention_mask is None))
        if not kwargs.get('min_length', None):
            raise KeyError
        if not kwargs.get('max_new_tokens', None) and not kwargs.get('max_length', None):
            raise KeyError
        kwargs['use_cache'] = True

        # bad_words_ids
        bad_words_ids = kwargs.get('bad_words_ids', [])
        for e in [self.toker.pad_token_id, self.toker.unk_token_id, self.toker.bos_token_id]:
            if e is not None and e != self.toker.eos_token_id:
                bad_words_ids.append([e])
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids.extend([[i] for i in range(self.toker.vocab_size, len(self.toker))])
        if bad_words_ids:
            kwargs['bad_words_ids'] = bad_words_ids

        # prepare the prefix ids for generation, and use prefix_length to truncate the generation output
        if self.config.is_encoder_decoder:
            if decoder_input_ids is not None:
                kwargs['decoder_input_ids'] = decoder_input_ids
                prefix_length = decoder_input_ids.size(1)
            else:
                prefix_length = 1 # removing bos
        else:
            prefix_length = input_ids.size(1) if input_ids is not None else 1

        # generation length
        kwargs['min_length'] = prefix_length + kwargs.get('min_length', 1)
        if kwargs.get('max_new_tokens', None):
            kwargs['max_length'] = prefix_length + kwargs['max_new_tokens']
            kwargs.pop('max_new_tokens')

        generations = self.custom_generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        if kwargs.get('return_dict_in_generate', False):
            generations.sequences = generations.sequences[:, prefix_length:]
            return generations
        else:
            return generations[:, prefix_length:]

    @torch.no_grad()
    def custom_generate(
        self,
        inputs=None,
        aux_inputs=None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
        **model_kwargs,
    ):
        # 0. get aux inputs
        aux_model_kwargs = {k[4:]: model_kwargs.pop(k) for k in sorted(model_kwargs.keys()) if k.startswith('aux_')}
        if len(aux_model_kwargs) == 0:
            aux_model_kwargs = copy.deepcopy(model_kwargs)
        assert sorted(aux_model_kwargs.keys()) == sorted(model_kwargs.keys()), (aux_model_kwargs.keys(), model_kwargs.keys())

        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        aux_inputs_tensor, _, aux_model_kwargs = self.antiexpert._prepare_model_inputs(aux_inputs, bos_token_id, aux_model_kwargs)
        aux_batch_size = aux_inputs_tensor.shape[0]

        expert_kwargs = copy.deepcopy(aux_model_kwargs)
        antiexpert_kwargs = copy.deepcopy(aux_model_kwargs)

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = expert_kwargs["output_attentions"] = antiexpert_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = expert_kwargs["output_hidden_states"] = antiexpert_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = expert_kwargs["use_cache"] = antiexpert_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )
            expert_kwargs["attention_mask"] = self.expert._prepare_attention_mask_for_generation(
                aux_inputs_tensor, pad_token_id, eos_token_id
            )
            antiexpert_kwargs["attention_mask"] = self.antiexpert._prepare_attention_mask_for_generation(
                aux_inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
            expert_kwargs = self.expert._prepare_encoder_decoder_kwargs_for_generation(
                aux_inputs_tensor, expert_kwargs, model_input_name
            )
            antiexpert_kwargs = self.antiexpert._prepare_encoder_decoder_kwargs_for_generation(
                aux_inputs_tensor, antiexpert_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )
            expert_input_ids = self.expert._prepare_decoder_input_ids_for_generation(
                aux_batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=expert_kwargs,
                device=aux_inputs_tensor.device,
            )
            antiexpert_input_ids = self.antiexpert._prepare_decoder_input_ids_for_generation(
                aux_batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=antiexpert_kwargs,
                device=aux_inputs_tensor.device,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor
            expert_input_ids = antiexpert_input_ids = inputs_tensor

        input_ids_seq_length = input_ids.shape[-1]
        # only bos_token_id
        assert antiexpert_input_ids.shape[-1] == antiexpert_input_ids.shape[-1] == input_ids_seq_length == 1

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length

        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but ``max_length`` is set to"
                f" {max_length}. This can lead to unexpected behavior. You should consider increasing"
                " ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None or force_words_ids is not None
        is_greedy_gen_mode = (
            (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and not is_constraint_gen_mode
        )
        is_sample_gen_mode = (
            (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and not is_constraint_gen_mode
        )
        is_beam_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and not is_constraint_gen_mode
        )
        is_beam_sample_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and not is_constraint_gen_mode
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and not is_constraint_gen_mode
        assert is_sample_gen_mode #or is_beam_gen_mode

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        # 9. go into different generation modes
        if is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
                renormalize_logits=renormalize_logits,
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            expert_input_ids, expert_kwargs = self.expert._expand_inputs_for_generation(
                expert_input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.expert.config.is_encoder_decoder,
                **expert_kwargs,
            )
            antiexpert_input_ids, antiexpert_kwargs = self.antiexpert._expand_inputs_for_generation(
                antiexpert_input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.antiexpert.config.is_encoder_decoder,
                **antiexpert_kwargs,
            )

            # 12. run sample
            return self.sample(
                input_ids,
                expert_input_ids=expert_input_ids,
                antiexpert_input_ids=antiexpert_input_ids,
                num_return_sequences=num_return_sequences,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                model_kwargs=model_kwargs,
                expert_kwargs=expert_kwargs,
                antiexpert_kwargs=antiexpert_kwargs,
            )

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

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # apply modification
            next_token_scores = next_token_scores + self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)

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
