import os
import numpy as np
import json
import multiprocessing as mp
import argparse
import csv


def is_negative(x):
    return x[0] > 0.5


def process(infer_data_path):
    #print('\t', infer_data_path)
    if not os.path.exists(f'{infer_data_path}/gen.txt'):
        return []

    stats_line = []

    first_gen = json.loads(open(f'{infer_data_path}/gen.txt').readline())['generation']
    if isinstance(first_gen, str):
        multiple = 1
    else:
        multiple = len(first_gen)

    if os.path.exists(f'{infer_data_path}/pred_list.txt'):
        preds = [json.loads(e) for e in open(f'{infer_data_path}/pred_list.txt')]
        preds = np.array(preds).reshape(-1, multiple, 2).transpose(2, 0, 1)
        #max_negative = preds[0].max(axis=-1).mean()
        emp_prob = (is_negative(preds).sum(axis=-1) > 0).mean()
        stats_line.extend([str(emp_prob)])
    else:
        stats_line.extend([''])

    if os.path.exists(f'{infer_data_path}/loss_large_list.txt'):
        losses = [json.loads(e) for e in open(f'{infer_data_path}/loss_large_list.txt')]
        ppls = [np.exp(e['loss'] / e['num_tokens']) for e in losses]
        ppl = np.mean(ppls)
        stats_line.append(ppl)
    else:
        stats_line.extend(['', ])

    if os.path.exists(f'{infer_data_path}/dist_list.txt'):
        dists = [json.loads(e) for e in open(f'{infer_data_path}/dist_list.txt')]
        dists = np.array(dists).mean(0)
        dists = [str(e) for e in dists][-2:]
        stats_line.extend(dists)
    else:
        stats_line.extend(['', '', ])

    return stats_line


def main(args):
    if not args.test:
        save_name = './results_bad.txt'
    else:
        save_name = './results_bad_test.txt'
    stats_writer = csv.writer(open(save_name, 'w'), delimiter='\t')

    pool = mp.Pool(mp.cpu_count())
    for stats_line in pool.imap(process, args.infer_data_paths):
        stats_writer.writerow(stats_line)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_data_paths', type=str, nargs='+')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    main(args)
