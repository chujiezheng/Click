import os
import numpy as np
import json
import argparse
import csv


def process(infer_data_path):
    print('\t', infer_data_path)
    stats_line = []

    if os.path.exists(f'{infer_data_path}/metric.json'):
        metric = json.load(open(f'{infer_data_path}/metric.json'))
        for metric_name in [
            #'dist-1',
            'ppl_macro',
            'acc', 'rep', 'wrep',
            'rep-2', 'rep-3',
            'diversity',
        ]:
            if metric_name in metric:
                stats_line.append(str(metric[metric_name]))
            else:
                stats_line.append('')

        if os.path.exists(f'{infer_data_path}/mauve.txt'):
            metric = open(f'{infer_data_path}/mauve.txt').readline().strip()
            stats_line.append(metric)
        else:
            stats_line.append('')

    return stats_line


def main(args):
    if not args.test:
        save_name = './results_wiki.txt'
    else:
        save_name = './results_wiki_test.txt'
    stats_writer = csv.writer(open(save_name, 'w'), delimiter='\t')

    for infer_data_path in args.infer_data_paths:
        stats_line = process(infer_data_path)
        stats_writer.writerow(stats_line)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_data_paths', type=str, nargs='+')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    main(args)
