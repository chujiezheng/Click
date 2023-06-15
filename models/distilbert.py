# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.distilbert.modeling_distilbert import (DistilBertConfig, DistilBertPreTrainedModel, DistilBertModel, DistilBertForSequenceClassification)
from utils.model_utils import BaseModel


class Model(BaseModel, DistilBertPreTrainedModel):
    def __init__(self, config: DistilBertConfig, num_labels):
        super().__init__(config)
        self.num_labels = config.num_labels = int(num_labels)
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_dict=None,
        validation=False,
        inference=False,
        **kwargs,
    ):
        assert (not validation and not inference) == self.training
        encoded_info = kwargs
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

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
