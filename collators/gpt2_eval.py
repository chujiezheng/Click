import json
from typing import List
import torch
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer


def collate_fn(data_list, toker: PreTrainedTokenizer, max_input_length=None, max_decoder_input_length=None, infer=False):
    features: List[Feature] = [convert_data_to_feature(e, toker, max_input_length, max_decoder_input_length, infer) for e in data_list]

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    if not infer:
        labels = torch.tensor([f.labels for f in features], dtype=torch.long)
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    else:
        references = [f.reference for f in features]
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'references': references,
        }

    return res


class Feature(object):
    def __init__(
        self,
        input_ids, attention_mask,
        labels,
        reference=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.reference = reference


def convert_data_to_feature(data, toker: PreTrainedTokenizer, max_input_length, max_decoder_input_length=None, infer=False) -> Feature:
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    source = process(data['source'])
    target = process(data['target'])
    eos = toker.eos_token_id
    reference = data['target']

    if infer:
        input_ids = source[-max_input_length:]
        attention_mask = [1.] * len(input_ids)
        input_ids = input_ids[:-1] + [eos] * (max_input_length - len(input_ids)) + input_ids[-1:]
        attention_mask = attention_mask[:-1] + [0.] * (max_input_length - len(attention_mask)) + attention_mask[-1:]
        labels = None
    else:
        source = source[-max_input_length:]
        input_ids = source + target[:-1][:max_decoder_input_length]
        labels = [-100] * (len(source) - 1) + target[:max_decoder_input_length + 1]
        attention_mask = [1.] * len(input_ids) + [0.] * (max_input_length + max_decoder_input_length - len(input_ids))
        input_ids = input_ids + [eos] * (max_input_length + max_decoder_input_length - len(input_ids))
        labels = labels + [-100] * (max_input_length + max_decoder_input_length - len(labels))

    feature = Feature(
        input_ids, attention_mask,
        labels,
        reference,
    )
    return feature
