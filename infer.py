import argparse
import json
import logging
import os
from importlib import import_module
import nltk
import numpy as np
from tqdm import tqdm
from functools import partial
from collections import defaultdict

from sklearn.metrics import classification_report, f1_score, confusion_matrix

import torch
from torch import Tensor
from transformers.trainer_utils import set_seed
from accelerate import Accelerator

from utils.building_utils import boolean_string, build_model
from utils.dataloader_utils import _norm, BatchDataLoader


def infer(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    assert len(args.infer_data_paths) == len(args.infer_names)

    accelerator = Accelerator()
    set_seed(args.seed)
    n_gpu = torch.cuda.device_count()

    #logger.info('Input Argument Information')
    #args_dict = vars(args)
    #for a in args_dict:
    #    logger.info('%-28s  %s' % (a, args_dict[a]))

    #########################################################################
    # Prepare Data Set
    ##########################################################################

    toker, model = build_model(args, checkpoint=args.load_checkpoint)
    model = accelerator.prepare(model)
    if n_gpu > 1:
        logger.info('use `torch.nn.DataParallel`')
        model = torch.nn.DataParallel(model)
    model.eval()

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Number of parameter = {}'.format(total_params))

    #########################################################################
    # Inference !
    ##########################################################################

    for infer_data_path, infer_name in zip(args.infer_data_paths, args.infer_names):
        if not os.path.exists(infer_data_path):
            logger.info(f'file {infer_data_path} does not exist')
            continue
        set_seed(args.seed)

        if args.save_path is not None:
            os.makedirs(args.save_path, exist_ok=True)
            save_path = f'{args.save_path}/{infer_name}'
            os.makedirs(save_path, exist_ok=True)
        else:
            assert args.load_checkpoint is not None
            checkpoint_dir_path = '/'.join(args.load_checkpoint.split('/')[:-1])
            save_path = f'{checkpoint_dir_path}/{infer_name}'
            os.makedirs(save_path, exist_ok=True)

        gen_file = os.path.join(save_path, 'gen.txt')
        gen_exist = os.path.exists(gen_file)
        metric_file = os.path.join(save_path, f'metric.json')
        metric_exist = os.path.exists(metric_file)

        if gen_exist and metric_exist:
            print('all have existed')
            continue
        elif gen_exist and args.only_infer:
            print('gen has existed while metric not required')
            continue

        metric_res = {}
        collate_fn = getattr(import_module('collators.' + args.collator_name), 'collate_fn')
        infer_dataloader = BatchDataLoader(
            data_path=infer_data_path,
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

        res = []
        other_res = defaultdict(list)
        for batch in tqdm(infer_dataloader, total=len(infer_dataloader), desc='inference', dynamic_ncols=True):
            labels = batch.pop('labels')
            batch['inference'] = True
            with torch.no_grad():
                encoded_info = model(**batch)
            if not args.only_infer:
                encoded_info['labels'] = labels

            for key in ['labels', 'preds', 'preds_top3', 'preds_dist']:
                if key in encoded_info:
                    encoded_info[key] = encoded_info[key].tolist()
                    other_res[key].extend(encoded_info[key])

            for idx in range(labels.size(0)):
                tmp_res_to_append = {}
                for key in ['labels', 'preds', 'preds_top3', 'preds_dist']:
                    if key in encoded_info:
                        tmp_res_to_append[key.replace('labels', 'label').replace('preds', 'pred')] = encoded_info[key][idx] if 'dist' not in key else ' '.join(map(str, encoded_info[key][idx]))

                res.append(tmp_res_to_append)

        with open(os.path.join(save_path, f'gen.txt'), 'w') as f:
            for line in res:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

        if not args.only_infer:
            metric_res_list = {}
            labels = np.array(other_res['labels'], dtype=int)
            preds = np.array(other_res['preds'], dtype=int)
            print(f'classification_report\t{save_path}\n', classification_report(labels, preds, digits=4))
            with open(os.path.join(save_path, f'confusion_matrix.json'), 'w') as f:
                json.dump(confusion_matrix(labels, preds).tolist(), f)
                #print('confusion_matrix\n', confusion_matrix(labels, preds))

            metric_res['acc'] = np.mean(labels == preds)
            metric_res['f1_micro'] = f1_score(labels, preds, average='micro')
            metric_res['f1_macro'] = f1_score(labels, preds, average='macro')
            metric_res_list['acc'] = (labels == preds).astype(int).tolist()

            if 'preds_top3' in other_res:
                preds_top3 = np.array(other_res['preds_top3'], dtype=int)
                metric_res['acc_top3'] = np.mean(np.sum((labels.reshape(-1, 1) - preds_top3) == 0, axis=-1) != 0)
                metric_res_list['acc_top3'] = (np.sum((labels.reshape(-1, 1) - preds_top3) == 0, axis=-1) != 0).astype(int).tolist()

            with open(os.path.join(save_path, f'metric.json'), 'w') as f:
                json.dump(metric_res, f, ensure_ascii=False, indent=2, sort_keys=True)

            if metric_res_list is not None:
                with open(os.path.join(save_path, f'metric_list.json'), 'w') as f:
                    json.dump(metric_res_list, f)


def main():
    #########################################################################
    # Prepare Parser
    ##########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--collator_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_args', type=str, nargs='+', default=[])

    parser.add_argument('--infer_data_paths', type=str, nargs='+', required=True)
    parser.add_argument('--infer_names', type=str, nargs='+', required=True)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument('--only_infer', action='store_true', help='do not conduct evaluations')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
