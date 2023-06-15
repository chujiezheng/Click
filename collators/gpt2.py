import json
from typing import List
import torch
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer


def collate_fn(data_list, toker: PreTrainedTokenizer, max_input_length=None, max_decoder_input_length=None, infer=False):
    features: List[Feature] = [convert_data_to_feature(e, toker, max_input_length, max_decoder_input_length) for e in data_list]

    assert not infer
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    labels = torch.tensor([f.labels for f in features], dtype=torch.long)

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    return res


class Feature(object):
    def __init__(
        self,
        input_ids, attention_mask,
        labels,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


def convert_data_to_feature(data, toker: PreTrainedTokenizer, max_input_length, max_decoder_input_length=None) -> Feature:
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    assert 'target' in data
    target = process(data['target'])
    eos = toker.eos_token_id

    input_ids = target[:-1][:max_input_length]
    labels = target[1:][:max_input_length]
    attention_mask = [1.] * len(input_ids) + [0.] * (max_input_length - len(input_ids))
    input_ids = input_ids + [eos] * (max_input_length - len(input_ids))
    labels = labels + [-100] * (max_input_length - len(labels))

    feature = Feature(
        input_ids, attention_mask,
        labels,
    )
    return feature
