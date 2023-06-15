# coding=utf-8
import logging

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)


class BaseModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.toker = None

    def tie_tokenizer_and_post_init(self, toker: PreTrainedTokenizer, process_index=0):
        # tying tokenizer is useful
        self.toker = toker
        old_num_tokens, _ = self.get_input_embeddings().weight.size()
        if len(self.toker) != old_num_tokens:
            self.resize_token_embeddings(len(self.toker))
            if process_index == 0:
                logger.info(f'resize token embeddings from {old_num_tokens} to {len(self.toker)}')
            self.resize_token_embeddings(len(self.toker))
            #self.init_new_tokens()
            self.init_new_tokens_with_semantic()
        self.init_new_layers()

    def init_new_tokens(self):
        # if we add new tokens, initialize them here
        pass

    def init_new_tokens_with_semantic(self):
        # we may need to initialize newly added tokens, with semantic initialization
        process = lambda x: self.toker.convert_tokens_to_ids(self.toker.tokenize(x))
        for i in range(self.toker.vocab_size, len(self.toker)):
            token = self.toker.convert_ids_to_tokens([i])[0]
            token = token[1:-1]
            ids = torch.LongTensor(process(token))
            embeds = torch.index_select(self.get_input_embeddings().weight.data.detach(), 0, ids)
            embeds = torch.mean(embeds, dim=0)
            self.get_input_embeddings().weight.data[i] = embeds
        self.tie_weights()

    def init_new_layers(self):
        # if we add new layers, initialize them here
        pass

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        **kwargs
    ):
        """
        (input_ids, attention_mask, decoder_input_ids)
        (input_ids, attention_mask)
        """
        assert not self.training
        assert self.toker is not None
        assert input_ids.size(0) == 1 or ((input_ids is None) == (attention_mask is None))
        if not kwargs.get('min_length', None):
            raise KeyError
        if not kwargs.get('max_new_tokens', None) and not kwargs.get('max_length', None):
            raise KeyError
        kwargs['use_cache'] = True

        # bad_words_ids
        bad_words_ids = kwargs.get('bad_words_ids', [])
        for e in [self.toker.pad_token_id, self.toker.unk_token_id, self.toker.bos_token_id]:
            if e is not None and e != self.toker.eos_token_id:
                bad_words_ids.append([e])
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids.extend([[i] for i in range(self.toker.vocab_size, len(self.toker))])
        if bad_words_ids:
            kwargs['bad_words_ids'] = bad_words_ids

        # prepare the prefix ids for generation, and use prefix_length to truncate the generation output
        if self.config.is_encoder_decoder:
            if decoder_input_ids is not None:
                kwargs['decoder_input_ids'] = decoder_input_ids
                prefix_length = decoder_input_ids.size(1)
            else:
                prefix_length = 1 # removing bos
        else:
            prefix_length = input_ids.size(1) if input_ids is not None else 1

        # generation length
        kwargs['min_length'] = prefix_length + kwargs.get('min_length', 1)
        if kwargs.get('max_new_tokens', None):
            kwargs['max_length'] = prefix_length + kwargs['max_new_tokens']
            kwargs.pop('max_new_tokens')

        # generate!
        generations = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        if kwargs.get('return_dict_in_generate', False):
            generations.sequences = generations.sequences[:, prefix_length:]
            return generations
        else:
            return generations[:, prefix_length:]
