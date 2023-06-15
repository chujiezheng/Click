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
    decoder_input_ids = torch.tensor([f.decoder_input_ids for f in features], dtype=torch.long)
    labels = torch.tensor([f.labels for f in features], dtype=torch.long)

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    }

    neg_decoder_input_ids = torch.tensor([e for f in features for e in f.neg_decoder_input_ids], dtype=torch.long)
    if len(neg_decoder_input_ids) == 0:
        return res

    pos_decoder_input_ids = torch.tensor([e for f in features for e in f.pos_decoder_input_ids], dtype=torch.long)
    pos_labels = torch.tensor([e for f in features for e in f.pos_labels], dtype=torch.long)
    neg_labels = torch.tensor([e for f in features for e in f.neg_labels], dtype=torch.long)
    selected_indices = torch.tensor([i for i, f in enumerate(features) for _ in range(len(f.neg_decoder_input_ids))], dtype=torch.long)

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,

        'pos_decoder_input_ids': pos_decoder_input_ids,
        'pos_labels': pos_labels,
        'neg_decoder_input_ids': neg_decoder_input_ids,
        'neg_labels': neg_labels,
        'selected_indices': selected_indices,
    }
    return res


class Feature(object):
    def __init__(
        self,
        input_ids, attention_mask,
        decoder_input_ids, labels,
        pos_decoder_input_ids, pos_labels,
        neg_decoder_input_ids, neg_labels,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.labels = labels

        self.pos_decoder_input_ids = pos_decoder_input_ids
        self.pos_labels = pos_labels
        self.neg_decoder_input_ids = neg_decoder_input_ids
        self.neg_labels = neg_labels


def convert_data_to_feature(data, toker: PreTrainedTokenizer, max_input_length, max_decoder_input_length=None) -> Feature:
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
    source = process(data['source'])
    target = process(data['target'])

    pad_token_id = toker.pad_token_id

    input_ids = source[-max_input_length:]
    decoder_input_ids = target[:-1][:max_decoder_input_length]
    labels = target[1:][:max_decoder_input_length]
    assert decoder_input_ids[1:] == labels[:-1]

    attention_mask = [1.] * len(input_ids) + [0.] * (max_input_length - len(input_ids))
    input_ids = input_ids + [pad_token_id] * (max_input_length - len(input_ids))
    decoder_input_ids = decoder_input_ids + [pad_token_id] * (max_decoder_input_length - len(decoder_input_ids))
    labels = labels + [-100] * (max_decoder_input_length - len(labels))

    pos_targets = [process(e) for e in data['pos_targets']]
    neg_targets = [process(e) for e in data['neg_targets']]

    pos_decoder_input_ids = [e[:-1][:max_decoder_input_length] for e in pos_targets]
    pos_labels = [e[1:][:max_decoder_input_length] for e in pos_targets]
    pos_decoder_input_ids = [e + [pad_token_id] * (max_decoder_input_length - len(e)) for e in pos_decoder_input_ids]
    pos_labels = [e + [-100] * (max_decoder_input_length - len(e)) for e in pos_labels]

    neg_decoder_input_ids = [e[:-1][:max_decoder_input_length] for e in neg_targets]
    neg_labels = [e[1:][:max_decoder_input_length] for e in neg_targets]
    neg_decoder_input_ids = [e + [pad_token_id] * (max_decoder_input_length - len(e)) for e in neg_decoder_input_ids]
    neg_labels = [e + [-100] * (max_decoder_input_length - len(e)) for e in neg_labels]

    feature = Feature(
        input_ids, attention_mask,
        decoder_input_ids, labels,
        pos_decoder_input_ids, pos_labels,
        neg_decoder_input_ids, neg_labels,
    )
    return feature
