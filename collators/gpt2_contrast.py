import json
from typing import List
import torch
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer


def collate_fn(data_list, toker: PreTrainedTokenizer, max_input_length=None, max_decoder_input_length=None, infer=False):
    features: List[Feature] = [convert_data_to_feature(e, toker, max_input_length, max_decoder_input_length) for e in data_list]

    assert not infer
    input_ids = torch.tensor([f.input_ids for f in features if f.input_ids is not None], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features if f.input_ids is not None], dtype=torch.float)
    labels = torch.tensor([f.labels for f in features if f.input_ids is not None], dtype=torch.long)

    if input_ids.size(0) == 0:
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.tensor([[0.]], dtype=torch.float)
        labels = torch.tensor([[-100]], dtype=torch.long)

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

    neg_input_ids = torch.tensor([e for f in features for e in f.neg_input_ids], dtype=torch.long)
    if len(neg_input_ids) == 0:
        return res

    pos_input_ids = torch.tensor([e for f in features for e in f.pos_input_ids], dtype=torch.long)
    pos_labels = torch.tensor([e for f in features for e in f.pos_labels], dtype=torch.long)
    neg_labels = torch.tensor([e for f in features for e in f.neg_labels], dtype=torch.long)

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,

        'pos_input_ids': pos_input_ids,
        'pos_labels': pos_labels,
        'neg_input_ids': neg_input_ids,
        'neg_labels': neg_labels,
    }
    return res


class Feature(object):
    def __init__(
        self,
        input_ids, attention_mask,
        labels,
        pos_input_ids, pos_labels,
        neg_input_ids, neg_labels,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

        self.pos_input_ids = pos_input_ids
        self.pos_labels = pos_labels
        self.neg_input_ids = neg_input_ids
        self.neg_labels = neg_labels


def convert_data_to_feature(data, toker: PreTrainedTokenizer, max_input_length, max_decoder_input_length=None) -> Feature:
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
    eos = toker.eos_token_id

    if 'target' in data:
        target = process(data['target'])

        input_ids = target[:-1][:max_input_length]
        labels = target[1:][:max_input_length]
        attention_mask = [1.] * len(input_ids) + [0.] * (max_input_length - len(input_ids))
        input_ids = input_ids + [eos] * (max_input_length - len(input_ids))
        labels = labels + [-100] * (max_input_length - len(labels))
    else:
        input_ids = None
        attention_mask = None
        labels = None

    pos_targets = [process(e) for e in data['pos_targets']]
    neg_targets = [process(e) for e in data['neg_targets']]

    # we use max_decoder_input_length as calibration data max length
    pos_input_ids = [e[:-1][:max_decoder_input_length] for e in pos_targets]
    pos_labels = [e[1:][:max_decoder_input_length] for e in pos_targets]
    pos_input_ids = [e + [eos] * (max_decoder_input_length - len(e)) for e in pos_input_ids]
    pos_labels = [e + [-100] * (max_decoder_input_length - len(e)) for e in pos_labels]

    neg_input_ids = [e[:-1][:max_decoder_input_length] for e in neg_targets]
    neg_labels = [e[1:][:max_decoder_input_length] for e in neg_targets]
    neg_input_ids = [e + [eos] * (max_decoder_input_length - len(e)) for e in neg_input_ids]
    neg_labels = [e + [-100] * (max_decoder_input_length - len(e)) for e in neg_labels]

    feature = Feature(
        input_ids, attention_mask,
        labels,
        pos_input_ids, pos_labels,
        neg_input_ids, neg_labels,
    )
    return feature
