# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import (RobertaConfig, RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead)
from utils.model_utils import BaseModel


class Model(BaseModel, RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, num_labels):
        super().__init__(config)
        self.num_labels = config.num_labels = int(num_labels)
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        return_dict=None,
        validation=False,
        inference=False,
        **kwargs,
    ):
        assert (not validation and not inference) == self.training
        encoded_info = kwargs
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = self.predict_label(logits, encoded_info)

        if inference:
            return encoded_info.copy()
        else:
            ppl_value = loss.new_tensor([0.], dtype=loss.dtype)
            if not validation:
                res = {
                    'all': loss,
                    'ppl': ppl_value,
                    'loss': loss,
                }
                return res
            else:
                return loss * input_ids.size(0), loss.new_tensor([input_ids.size(0)])

    def predict_label(self, logits, encoded_info):
        preds = torch.argmax(logits, dim=-1)
        if self.num_labels > 3:
            preds_top3 = torch.topk(logits, k=3, dim=-1)[1]
        loss = None
        if 'labels' in encoded_info:
            loss = F.cross_entropy(logits, encoded_info.get('labels'), reduction='mean')
        encoded_info['preds'] = preds
        if self.num_labels > 3:
            encoded_info['preds_top3'] = preds_top3
        encoded_info['preds_dist'] = F.softmax(logits, dim=-1)
        return loss
