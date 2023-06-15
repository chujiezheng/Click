import argparse
import json
import logging
import os
from importlib import import_module
import nltk
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import multiprocessing as mp
from collections import Counter

import torch
from torch import Tensor
from transformers.trainer_utils import set_seed
from accelerate import Accelerator

from metrics import Metrics


def _norm(x):
    return " ".join(x.strip().split())


def tokenize(x):
    return nltk.word_tokenize(_norm(x))


def calculate_distinct(generation):
    generation = [tokenize(e) for e in generation]
    lengths = [len(e) for e in generation]
    distinct = []
    for n in range(1, 4):
        ngrams = []
        for g in generation:
            tmp_ngrams = list(zip(*[g[i:] for i in range(n)]))
            ngrams.extend(tmp_ngrams)
        ngrams = Counter(ngrams)
        #distinct.append(len(ngrams) / sum(ngrams.values()))
        distinct.append(len(ngrams) / sum(lengths))
    return distinct


def evaluate(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    pool = mp.Pool(mp.cpu_count() * 2)

    for infer_data_path in args.infer_data_paths:
        generation_file = infer_data_path + '/gen.txt'
        if not os.path.exists(generation_file):
            continue
        if os.path.exists(f'{infer_data_path}/dist_list.txt'):
            print('prediction have existed')
            continue

        generations = [json.loads(e)['generation'] for e in open(generation_file)]
        distinct = [e for e in pool.imap(calculate_distinct, tqdm(generations, dynamic_ncols=True, total=len(generations)))]

        with open(infer_data_path + '/dist_list.txt', 'w') as f:
            for d in distinct:
                f.write(json.dumps(d) + '\n')


def main():
    #########################################################################
    # Prepare Parser
    ##########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--context_file', type=str, required=True)
    parser.add_argument('--infer_data_paths', type=str, nargs='+')

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
