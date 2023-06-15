# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2 import (GPT2LMHeadModel, GPT2Config)
from transformers.models.gpt_neo import (GPTNeoForCausalLM, GPTNeoConfig)
from transformers.models.gptj import (GPTJForCausalLM, GPTJConfig)
from transformers.models.opt import (OPTForCausalLM, OPTConfig)
from utils.model_utils import BaseModel


class Model(BaseModel, GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
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
        masked_lm_loss = None
        if self.training or validation:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_mask = labels.ne(-100).type_as(loss)
            label_size = label_mask.sum(1).type_as(loss)
            masked_lm_loss = loss.sum() / torch.clamp(label_size.sum(), min=1e-5)
            ppl_value = masked_lm_loss.exp()

            if validation:
                preds = torch.argmax(lm_logits, dim=-1)
                acc = ((preds == labels) * label_mask).type_as(loss) # [bs, length]
                occurrence = torch.tril(preds.unsqueeze(-1) == labels.unsqueeze(-2), -1) # [bs, length, length]
                rep = (occurrence.sum(dim=-1) > 0).type_as(loss) * label_mask # [bs, length]
                wrep = rep * (1. - acc)

        outputs.loss = masked_lm_loss

        if not self.training and not validation: # inference
            return outputs
        elif self.training: # training
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res
        else: # validation
            assert not self.training
            return loss, label_size, acc, rep, wrep
