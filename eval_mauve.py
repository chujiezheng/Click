import argparse
import json
import logging
import os
from importlib import import_module
import numpy as np
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
from collections import Counter
import nltk
from mauve import compute_mauve
from transformers import AutoTokenizer


def infer(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    gpt2_toker = AutoTokenizer.from_pretrained('/home/zhengchujie/pretrained-models/gpt2-small', use_fast=True)

    contexts = []
    p_text = []
    for e in open(args.context_file):
        e = json.loads(e)
        contexts.append(gpt2_toker.decode(gpt2_toker.encode(e['source'])))
        p_text.append(contexts[-1] + e['target'])

    p_features = None

    if args.save_name == 'large':
        featurize_model_name = f'/home/zhengchujie/pretrained-models-large/gpt2-large'
    else:
        featurize_model_name = f'/home/zhengchujie/pretrained-models/gpt2-{args.save_name}'

    for infer_data_path in args.infer_data_paths:
        generation_file = infer_data_path + '/gen.txt'
        if not os.path.exists(generation_file):
            continue
        if os.path.exists(infer_data_path + f'/mauve_{args.save_name}.txt'):
            print('prediction have existed')
            continue
        q_text = []
        lines = open(generation_file).readlines()
        assert len(lines) == len(contexts)
        for idx, e in enumerate(lines):
            g = json.loads(e)['generation']
            q_text.append(contexts[idx] + g)

        if p_features is None:
            out = compute_mauve(
                p_text=p_text,
                q_text=q_text,
                max_text_length=160,
                featurize_model_name=featurize_model_name,
                batch_size=args.batch_size,
                verbose=False
            )
            p_features = out.p_features
        else:
            out = compute_mauve(
                p_features=p_features,
                q_text=q_text,
                max_text_length=160,
                featurize_model_name=featurize_model_name,
                batch_size=args.batch_size,
                verbose=False
            )

        res = out.mauve
        with open(infer_data_path + f'/mauve_{args.save_name}.txt', 'w') as f:
            f.write(str(res) + '\n')


def main():
    #########################################################################
    # Prepare Parser
    ##########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_data_paths', type=str, nargs='+')
    parser.add_argument('--save_name', type=str, required=True, choices=['small', 'medium', 'large'])
    parser.add_argument('--context_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    args = parser.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
