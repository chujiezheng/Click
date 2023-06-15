import argparse
import json
import logging
import os
from importlib import import_module
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
from torch import Tensor
from transformers.trainer_utils import set_seed
from accelerate import Accelerator

from utils.building_utils import boolean_string, build_model
from utils.dataloader_utils import _norm, BatchDataLoader


def _norm(x):
    return " ".join(x.strip().split())


def make_sample(args, toker, context, generation):
    utterances = context + [generation]
    dialogue_context = ''
    for i, utterance in enumerate(utterances):
        text = _norm(utterance)
        if i % 2 == 0:
            dialogue_context += f'Human: {text}\n'
        else:
            dialogue_context += f'Bot: {text}\n'

    dialogue_context = dialogue_context.strip()
    dialogue_context = toker.convert_tokens_to_string(toker.tokenize(dialogue_context)[-args.max_input_length+2:])
    d = {
        'text': dialogue_context,
        'label': 0,
    }
    return d


def run(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    accelerator = Accelerator()
    set_seed(args.seed)
    n_gpu = torch.cuda.device_count()

    toker, model = build_model(args)
    model = accelerator.prepare(model)
    if n_gpu > 1:
        logger.info('use `torch.nn.DataParallel`')
        model = torch.nn.DataParallel(model)
    model.eval()

    contexts = [json.loads(e) for e in open(args.context_file)]
    contexts = [e['context'] for e in contexts]
    for infer_data_path in args.infer_data_paths:
        generation_file = infer_data_path + '/gen.txt'
        if not os.path.exists(generation_file):
            continue
        if os.path.exists(infer_data_path + '/pred_list.txt'):
            print('prediction have existed')
            continue
        generations = [json.loads(e)['generation'] for e in open(generation_file)]
        assert len(contexts) == len(generations)

        data_list = []
        for context, generation in tqdm(zip(contexts, generations), total=len(contexts), ncols=0, leave=False):
            if not isinstance(generation, list):
                generation = [generation]
            for g in generation:
                tmp_data = make_sample(args, toker, context, g)
                data_list.append(tmp_data)

        collate_fn = getattr(import_module('collators.' + args.collator_name), 'collate_fn')
        infer_dataloader = BatchDataLoader(
            data_list=data_list,
            batch_size=args.batch_size,
            collate_fn=partial(
                collate_fn,
                toker=toker,
                max_input_length=args.max_input_length,
                infer=True,
            ),
            shuffle=False,
        )
        infer_dataloader = accelerator.prepare(infer_dataloader)

        preds = []
        for batch in tqdm(infer_dataloader, total=len(infer_dataloader), desc='inference', dynamic_ncols=True):
            batch.pop('labels')
            batch['inference'] = True
            with torch.no_grad():
                encoded_info = model(**batch)
            preds.extend(encoded_info['preds_dist'].tolist())

        with open(infer_data_path + '/pred_list.txt', 'w') as f:
            for d in preds:
                f.write(json.dumps(d) + '\n')


def main():
    #########################################################################
    # Prepare Parser
    ##########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--collator_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--model_args', type=str, nargs='+', default=[])

    parser.add_argument('--context_file', type=str, required=True)
    parser.add_argument('--infer_data_paths', type=str, nargs='+')
    parser.add_argument("--max_input_length", type=int, default=48)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
