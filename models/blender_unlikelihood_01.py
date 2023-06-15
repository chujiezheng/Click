# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.blenderbot.modeling_blenderbot import (BlenderbotConfig, BlenderbotForConditionalGeneration,)
from utils.model_utils import BaseModel

GAMMA = 0.1


class Model(BaseModel, BlenderbotForConditionalGeneration):
    def __init__(self, config: BlenderbotConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        cls_labels=None,
        use_cache=None,
        return_dict=True,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        assert not (self.training and validation)
        if self.training or validation:
            assert labels is not None
            assert cls_labels is not None
            cls_labels = cls_labels.unsqueeze(-1).expand_as(labels).contiguous()
            use_cache = False

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
            loss = loss.view(labels.size(0), labels.size(1)) * cls_labels
            label_size = (labels.ne(-100) * cls_labels).sum(1).type_as(loss)
            masked_lm_loss = loss.sum() / torch.clamp(label_size.sum(), min=1e-5)
            ppl_value = masked_lm_loss.exp()

            negative_lm_logits = torch.clamp(1. - F.softmax(lm_logits, dim=-1), min=1e-5).log()
            negative_loss = F.cross_entropy(negative_lm_logits.view(-1, negative_lm_logits.size(-1)), labels.view(-1), reduction='none')
            negative_loss = negative_loss.view(labels.size(0), labels.size(1)) * (1. - cls_labels)
            negative_label_size = (labels.ne(-100) * (1. - cls_labels)).sum(1).type_as(loss)
            negative_lm_loss = negative_loss.sum() / negative_label_size.sum()

        outputs.loss = masked_lm_loss

        if not self.training and not validation: # inference
            return outputs
        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss + GAMMA * negative_lm_loss, 'ppl': ppl_value, }
            return res
        else: # validation
            assert not self.training
            return loss, label_size
