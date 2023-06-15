# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.distributions import Categorical
from transformers.models.blenderbot.modeling_blenderbot import (BlenderbotConfig, BlenderbotForConditionalGeneration,)
from utils.model_utils import BaseModel

GAMMA = 0.2


class CringeLoss(CrossEntropyLoss):
    def __init__(self, k=5, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def __call__(self, x, y, classifier_labels, **kwargs):

        # Compute the CrossEntropy loss for the positive labels and mask
        # with classifier labels to not train with negative feedback (0)
        ce_loss = super().__call__(x, y, **kwargs)
        ce_loss *= classifier_labels

        # compute the contrastive loss part for the negative labels
        # first, get the positives as the top predictions != target
        preds = torch.topk(x, k=self.k + 1, axis=-1)
        y_rep = y.unsqueeze(1).repeat(1, self.k + 1)
        logits = preds.values - (preds.indices == y_rep) * 1e10

        # if the positive is not in the first k predictions, mask out
        # the final (k+1)'s logit
        prediction_mask = torch.cat(
            (torch.zeros_like(logits)[:, :-1],
            torch.abs((preds.indices == y_rep).sum(-1).unsqueeze(1) - 1),),
            1,)
        logits -= prediction_mask * 1e10

        # Sample from the categorical distribution of the top-k predictions
        # (with the label masked out).
        preds_dist = Categorical(logits=logits)
        idx_sample = preds_dist.sample()
        sample_preds_values = preds.values[torch.arange(x.shape[0]), idx_sample]

        # Concatenate the logits of the preds with the negative label's logits.
        x_negative_target = x[torch.arange(x.shape[0]), y]
        x_cr = torch.concat(
            [x_negative_target.unsqueeze(1), sample_preds_values.unsqueeze(1)], -1)

        # Create the y's for the x_cr (the correct label is always index 1).
        y_cr = torch.ones(y.shape).type(y.dtype).to(x_cr.device)

        # Compute the Cringe loss as cross entropy loss between x_cr, y_cr
        # and mask out the positive labels.
        cr_loss = super().__call__(x_cr, y_cr, **kwargs)

        return ce_loss, cr_loss


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
            loss, negative_loss = CringeLoss(reduction='none')(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), cls_labels.view(-1))
            loss = loss.view(labels.size(0), labels.size(1)) * cls_labels
            label_size = (labels.ne(-100) * cls_labels).sum(1).type_as(loss)
            masked_lm_loss = loss.sum() / torch.clamp(label_size.sum(), min=1e-5)
            ppl_value = masked_lm_loss.exp()

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
