# coding=utf-8

import json
import logging
import os

from importlib import import_module
import torch
from transformers import (AutoTokenizer, AutoConfig)

logger = logging.getLogger(__name__)


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def build_model(args, only_toker=False, checkpoint=None, process_index=-1):
    # blenderbot tokenizer would add a mask token by default, so we abandon it
    if 'blenderbot-' in args.pretrained_model_path:
        toker = AutoTokenizer.from_pretrained(args.pretrained_model_path, mask_token=None, use_fast=False)
    else:
        toker = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)
    if only_toker:
        return toker

    # import the model from ``models''
    Model = getattr(import_module('models.' + args.model_name), 'Model')
    model = Model.from_pretrained(args.pretrained_model_path, *args.model_args)
    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.tie_tokenizer_and_post_init(toker, process_index)

    if checkpoint is not None and os.path.exists(checkpoint):
        if process_index == 0:
            logger.info('loading finetuned model from %s' % checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')), strict=False)

    return toker, model
