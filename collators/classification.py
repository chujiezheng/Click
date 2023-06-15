import json
from typing import List
import torch
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import logging

logging.set_verbosity_error()


def collate_fn(data_list, toker: PreTrainedTokenizer, max_input_length=None, max_decoder_input_length=None, infer=False):
    features: List[Feature] = [convert_data_to_feature(e, toker, max_input_length, max_decoder_input_length) for e in data_list]

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    labels = torch.tensor([f.label for f in features], dtype=torch.long)

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels, # [bs] for single-label [bs, label_num] for multi-label
    }
    return res


# `Feature` is a class that contains all the information needed to train a model
class Feature(object):
    def __init__(
        self, input_ids,
        attention_mask, token_type_ids, label,
        input_len,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.input_len = input_len


def convert_data_to_feature(data, toker: PreTrainedTokenizer, max_input_length, max_decoder_input_length=None) -> Feature:
    processed_data = toker(
        data['text'],
        data['text2'] if 'text2' in data else None,
        padding='max_length',
        truncation='longest_first',
        max_length=max_input_length,
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    feature = Feature(
        processed_data['input_ids'],
        processed_data['attention_mask'],
        processed_data['token_type_ids'],
        data['label'],
        len([e for e in processed_data['input_ids'] if e != toker.pad_token_id])
    )
    return feature

