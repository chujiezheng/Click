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


def nltk_repetition(text):
    tokens = nltk.word_tokenize(text)
    if len(tokens) <= 4:
        tokens = [e for e in list(text) if e.strip()]
    repn = {}
    for k in range(2, 5):
        ngrams = list(zip(*[tokens[i:] for i in range(k)]))
        ngrams = Counter(ngrams)
        repn[k] = 1. - len(ngrams) / max(sum(ngrams.values()), 1e-10)
    div = (1. - repn[2]) * (1. - repn[3]) * (1. - repn[4])
    return div


def infer(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    pool = mp.Pool(mp.cpu_count() * 2)

    contexts = []
    for e in open(args.context_file):
        e = json.loads(e)
        contexts.append(e['source'])

    for infer_data_path in args.infer_data_paths:
        generation_file = infer_data_path + '/gen.txt'
        if not os.path.exists(generation_file):
            continue
        if os.path.exists(infer_data_path + '/div_list.txt'):
            print('prediction have existed')
            #continue
        generations = []
        lines = open(generation_file).readlines()
        assert len(lines) == len(contexts)
        for idx, e in enumerate(lines):
            g = json.loads(e)['generation']
            if not isinstance(g, list):
                g = [g]
            for gg in g:
                #generations.append(contexts[idx] + gg)
                generations.append(gg)
        preds_new = [e for e in pool.imap(nltk_repetition, tqdm(generations, total=len(generations), ncols=0))]

        with open(infer_data_path + '/div_list.txt', 'w') as f:
            for d in preds_new:
                f.write(json.dumps(d) + '\n')

    pool.close()


def main():
    #########################################################################
    # Prepare Parser
    ##########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_data_paths', type=str, nargs='+')
    parser.add_argument('--context_file', type=str, required=True)

    args = parser.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
