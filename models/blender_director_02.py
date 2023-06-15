# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.blenderbot.modeling_blenderbot import (BlenderbotConfig, BlenderbotForConditionalGeneration,)
from utils.model_utils import BaseModel
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)

GAMMA = 0.2


class Model(BaseModel, BlenderbotForConditionalGeneration):
    def __init__(self, config: BlenderbotConfig, alpha='1.0'):
        super().__init__(config)
        self.cls_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self._init_weights(self.cls_head)
        self.alpha = float(alpha)

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
        if self.training:
            assert labels is not None
            assert cls_labels is not None
            cls_labels = cls_labels.unsqueeze(-1).expand(*labels.size()).contiguous()
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        cls_logits = self.cls_head(outputs[0])

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

            if self.training:
                cls_logits = cls_logits.view(-1, self.model.shared.num_embeddings)
                cls_tgt_logits = cls_logits[range(cls_logits.size(0)), labels.view(-1)] # [bs * tgt_len]
                cls_loss = F.binary_cross_entropy_with_logits(cls_tgt_logits, cls_labels.view(-1), reduction='none')
                cls_loss = cls_loss.view(labels.size(0), labels.size(1)) * labels.ne(-100) # [bs, tgt_len]
                cls_loss = cls_loss.sum() / labels.ne(-100).sum()

        else:
            cls_logits = cls_logits.view(lm_logits.size(0), -1, self.model.shared.num_embeddings)
            lm_logits = torch.log_softmax(lm_logits, dim=-1) + torch.sigmoid(cls_logits) * self.alpha

        outputs = Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

        if not self.training and not validation: # inference
            return outputs
        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss + GAMMA * cls_loss, 'ppl': ppl_value, 'cls_loss': cls_loss, }
            return res
        else: # validation
            assert not self.training
            return loss, label_size, acc, rep, wrep
