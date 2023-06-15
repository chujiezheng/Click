# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2 import (GPT2LMHeadModel, GPT2Config)
from transformers.models.gpt_neo import (GPTNeoForCausalLM, GPTNeoConfig)
from transformers.models.gptj import (GPTJForCausalLM, GPTJConfig)
from transformers.models.opt import (OPTForCausalLM, OPTConfig)
from utils.model_utils import BaseModel

GAMMA = 0.1


class Model(BaseModel, GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, alpha):
        super().__init__(config)
        self.alpha = float(alpha)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pos_input_ids=None,
        neg_input_ids=None,
        past_key_values=None,
        labels=None,
        pos_labels=None,
        neg_labels=None,
        use_cache=None,
        return_dict=True,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        assert self.training and not validation
        assert labels is not None
        use_cache = False

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=None,
            return_dict=return_dict,
            use_cache=use_cache,
            **kwargs
        )

        lm_logits = outputs.logits
        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
        loss = loss.view(labels.size(0), labels.size(1))
        label_size = labels.ne(-100).sum(1).type_as(loss)
        masked_lm_loss = loss.sum() / torch.clamp(label_size.sum(), min=1e-5)
        ppl_value = masked_lm_loss.exp()

        outputs.loss = masked_lm_loss

        if neg_input_ids is None:
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res

        # alpha
        pos_outputs = super().forward(
            input_ids=pos_input_ids,
            attention_mask=None,
            labels=None,
            return_dict=return_dict,
            use_cache=use_cache,
            **kwargs
        )
        pos_lm_logits = pos_outputs.logits
        pos_loss = F.cross_entropy(pos_lm_logits.view(-1, pos_lm_logits.size(-1)), pos_labels.view(-1), reduction='none')
        pos_loss = pos_loss.view(pos_labels.size(0), pos_labels.size(1)).sum(-1)

        neg_outputs = super().forward(
            input_ids=neg_input_ids,
            attention_mask=None,
            labels=None,
            return_dict=return_dict,
            use_cache=use_cache,
            **kwargs
        )
        neg_lm_logits = neg_outputs.logits
        neg_loss = F.cross_entropy(neg_lm_logits.view(-1, neg_lm_logits.size(-1)), neg_labels.view(-1), reduction='none')
        neg_loss = neg_loss.view(neg_labels.size(0), neg_labels.size(1)).sum(-1)

        # we have pos_loss < neg_loss
        loss1 = torch.clamp(self.alpha + pos_loss - neg_loss, min=0.)
        loss1 = loss1.mean()

        res = {'all': masked_lm_loss + GAMMA * loss1, 'ppl': ppl_value, 'loss': masked_lm_loss, 'loss1': loss1, }
        return res
