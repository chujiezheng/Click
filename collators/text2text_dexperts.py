import json
from typing import List
import torch
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

MAX_AUX_LENGTH = 32


def collate_fn(data_list, toker: PreTrainedTokenizer, max_input_length=None, max_decoder_input_length=None, infer=False):
    features: List[Feature] = [convert_data_to_feature(e, toker, max_input_length, max_decoder_input_length) for e in data_list]

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    aux_input_ids = torch.tensor([f.aux_input_ids for f in features], dtype=torch.long)
    aux_attention_mask = torch.tensor([f.aux_attention_mask for f in features], dtype=torch.float)
    if not infer:
        decoder_input_ids = torch.tensor([f.decoder_input_ids for f in features], dtype=torch.long)
        labels = torch.tensor([f.labels for f in features], dtype=torch.long)
    else:
        decoder_input_ids = None
        references = None
        if all(f.reference is not None for f in features):
            references = [f.reference for f in features]

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'aux_input_ids': aux_input_ids,
        'aux_attention_mask': aux_attention_mask,
        'decoder_input_ids': decoder_input_ids,
    }
    if not infer:
        res['labels'] = labels
    elif references is not None:
        res['references'] = references
    return res


class Feature(object):
    def __init__(
        self,
        input_ids, attention_mask,
        aux_input_ids, aux_attention_mask,
        decoder_input_ids, labels,
        reference=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.aux_input_ids = aux_input_ids
        self.aux_attention_mask = aux_attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.labels = labels
        self.reference = reference


def convert_data_to_feature(data, toker: PreTrainedTokenizer, max_input_length, max_decoder_input_length=None) -> Feature:
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
    source = process(data['source'])
    target = process(data['target'])
    reference = toker.decode(target[1:-1], skip_special_tokens=True)
    if 'aux_source' in data:
        aux_source = process(data['aux_source'])
        max_aux_length = MAX_AUX_LENGTH
    else:
        aux_source = source[:]
        max_aux_length = max_input_length

    pad_token_id = toker.pad_token_id

    #assert len(source) <= max_input_length
    #assert len(target) - 1 <= max_decoder_input_length
    input_ids = source[-max_input_length:]
    aux_input_ids = aux_source[:max_aux_length]
    decoder_input_ids = target[:-1][:max_decoder_input_length]
    labels = target[1:][:max_decoder_input_length]
    assert decoder_input_ids[1:] == labels[:-1]

    attention_mask = [1.] * len(input_ids) + [0.] * (max_input_length - len(input_ids))
    input_ids = input_ids + [pad_token_id] * (max_input_length - len(input_ids))
    aux_attention_mask = [1.] * len(aux_input_ids) + [0.] * (max_aux_length - len(aux_input_ids))
    aux_input_ids = aux_input_ids + [pad_token_id] * (max_aux_length - len(aux_input_ids))
    decoder_input_ids = decoder_input_ids + [pad_token_id] * (max_decoder_input_length - len(decoder_input_ids))
    labels = labels + [-100] * (max_decoder_input_length - len(labels))

    feature = Feature(
        input_ids, attention_mask,
        aux_input_ids, aux_attention_mask,
        decoder_input_ids, labels,
        reference,
    )
    return feature
