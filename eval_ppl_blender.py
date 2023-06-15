
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
from utils.eval_utils import eval_model_loss


def _norm(x):
    return " ".join(x.strip().split())


def make_source(toker, x):
    if isinstance(x, list):
        x = [' ' + e.strip() for e in x]
        x = '  '.join(x) + toker.eos_token
    return x


def make_sample(toker, context, generation):
    target = toker.bos_token + ' ' + generation.strip() + toker.eos_token
    d = {'source': context, 'target': target}
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
    contexts = [make_source(toker, e['context'] if 'context' in e else e['source']) for e in contexts]
    for infer_data_path in args.infer_data_paths:
        generation_file = infer_data_path + '/gen.txt'
        if not os.path.exists(generation_file):
            continue
        if os.path.exists(f'{infer_data_path}/loss_{args.save_name}_list.txt'):
            print('prediction have existed')
            continue
        generations = [json.loads(e)['generation'] for e in open(generation_file)]
        assert len(contexts) == len(generations)

        data_list = []
        for context, generation in tqdm(zip(contexts, generations), total=len(contexts), ncols=0, leave=False):
            if not isinstance(generation, list):
                generation = [generation]
            for g in generation:
                tmp_data = make_sample(toker, context, g)
                data_list.append(tmp_data)

        collate_fn = getattr(import_module('collators.' + args.collator_name), 'collate_fn')
        eval_dataloader = BatchDataLoader(
            data_list=data_list,
            batch_size=args.batch_size,
            collate_fn=partial(
                collate_fn,
                toker=toker,
                max_input_length=args.max_input_length,
                max_decoder_input_length=args.max_decoder_input_length,
                infer=False,
            ),
            shuffle=False,
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        _, eval_ppl_micro, *_, pointwise_loss, pointwise_sample = eval_model_loss(
            accelerator=accelerator,
            model=model,
            eval_dataloader=eval_dataloader,
            epoch_id=0,
            infer=True,
        )
        eval_loss_list = [{'loss': np.sum(x), 'num_tokens': y} for x, y in zip(pointwise_loss, pointwise_sample)]
        with open(f'{infer_data_path}/loss_{args.save_name}_list.txt', 'w') as f:
            for d in eval_loss_list:
                f.write(json.dumps(d) + '\n')


def main():
    #########################################################################
    # Prepare Parser
    ##########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--collator_name', type=str, default='text2text')
    parser.add_argument('--save_name', type=str, required=True, choices=['self', 'large'])
    parser.add_argument('--model_name', type=str, default='blender')
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--model_args', type=str, nargs='+', default=[])

    parser.add_argument('--context_file', type=str, required=True)
    parser.add_argument('--infer_data_paths', type=str, nargs='+')
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_decoder_input_length", type=int, default=24)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
