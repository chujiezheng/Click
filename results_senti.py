import os
import numpy as np
import json
import multiprocessing as mp
import argparse
import csv
from functools import partial


def calculate_positive_ratio(file_path):
    preds = [json.loads(e) for e in open(file_path)]
    preds = np.array(preds).transpose(1, 0)
    mean = (preds[0] < 0.5).mean() * 100
    return mean


def process(infer_data_path, key):
    #print('\t', infer_data_path)
    if not (os.path.exists(f'{infer_data_path}/neutral/gen.txt') and
            os.path.exists(f'{infer_data_path}/{key}/gen.txt')):
        return []

    stats_line = []
 
    if key == 'positive':
        if os.path.exists(f'{infer_data_path}/positive/pred_list.txt'):
            value = 100. - calculate_positive_ratio(f'{infer_data_path}/positive/pred_list.txt')
            stats_line.append(str(value))
        else:
            stats_line.append('')
    else:
        if os.path.exists(f'{infer_data_path}/negative/pred_list.txt'):
            value = calculate_positive_ratio(f'{infer_data_path}/negative/pred_list.txt')
            stats_line.append(str(value))
        else:
            stats_line.append('')

    if os.path.exists(f'{infer_data_path}/neutral/pred_list.txt'):
        value = calculate_positive_ratio(f'{infer_data_path}/neutral/pred_list.txt')
        if key == 'positive':
            value = 100. - value
        stats_line.append(str(value))
    else:
        stats_line.append('')

    if os.path.exists(f'{infer_data_path}/neutral/loss_large_list.txt') and os.path.exists(f'{infer_data_path}/{key}/loss_large_list.txt'):
        losses = []
        if os.path.exists(f'{infer_data_path}/neutral/loss_large_list.txt'):
            losses += [json.loads(e) for e in open(f'{infer_data_path}/neutral/loss_large_list.txt')]
        if os.path.exists(f'{infer_data_path}/{key}/loss_large_list.txt'):
            losses += [json.loads(e) for e in open(f'{infer_data_path}/{key}/loss_large_list.txt')]
        ppls = [np.exp(e['loss'] / e['num_tokens']) for e in losses]
        ppls = [e for e in ppls if e < 1e4]
        ppl = np.mean(ppls)
        #ppl = np.exp(np.sum([e['loss'] for e in losses]) / np.sum([e['num_tokens'] for e in losses]))
        stats_line.append(ppl)
    else:
        stats_line.append('')

    if os.path.exists(f'{infer_data_path}/neutral/dist_list.txt') and os.path.exists(f'{infer_data_path}/{key}/dist_list.txt'):
        dists = []
        if os.path.exists(f'{infer_data_path}/neutral/dist_list.txt'):
            dists += [json.loads(e) for e in open(f'{infer_data_path}/neutral/dist_list.txt')]
        if os.path.exists(f'{infer_data_path}/{key}/dist_list.txt'):
            dists += [json.loads(e) for e in open(f'{infer_data_path}/{key}/dist_list.txt')]
        dists = np.array(dists).mean(0)
        dists = [str(e) for e in dists][-2:]
        stats_line.extend(dists)
    else:
        stats_line.extend([ '', '', ])

    return stats_line


def main(args):
    if args.positive:
        save_name = './results_senti_pos.txt'
    else:
        save_name = './results_senti_neg.txt'
    stats_writer = csv.writer(open(save_name, 'w'), delimiter='\t')

    pool = mp.Pool(mp.cpu_count())
    for stats_line in pool.imap(
        partial(process, key='negative' if args.positive else 'positive'),
        args.infer_data_paths
    ):
        stats_writer.writerow(stats_line)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_data_paths', type=str, nargs='+')
    parser.add_argument('--positive', action='store_true')
    parser.add_argument('--negative', action='store_true')

    args = parser.parse_args()

    assert args.positive or args.negative
    assert not (args.positive and args.negative)

    main(args)
