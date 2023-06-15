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

import torch
from torch import Tensor
from transformers.trainer_utils import set_seed
from accelerate import Accelerator

from utils.building_utils import boolean_string, build_model
from utils.eval_utils import eval_model_loss
from utils.dataloader_utils import _norm, BatchDataLoader


def cut_sequence_to_eos(seq, eos_token_id):
    ret = []
    for t in seq:
        if len(ret) > 0 and t == eos_token_id:
            break
        ret.append(t)
    return ret


def cut_label_to_golden(seq):
    ret = []
    for t in seq:
        if t == -100:
            if len(ret) == 0:
                continue
            else:
                break
        ret.append(t)
    return ret


def generate(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    assert len(args.infer_data_paths) == len(args.infer_names)
    if args.only_evaluate:
        args.only_generate = False

    accelerator = Accelerator()
    set_seed(args.seed)

    #logger.info('Input Argument Information')
    #args_dict = vars(args)
    #for a in args_dict:
    #    logger.info('%-28s  %s' % (a, args_dict[a]))

    #########################################################################
    # Prepare Data Set
    ##########################################################################

    toker, model = build_model(args, checkpoint=args.load_checkpoint)
    model = accelerator.prepare(model)
    model.eval()

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Number of parameter = {}'.format(total_params))

    eos_token_id = toker.eos_token_id # if model.config.is_encoder_decoder else toker.convert_tokens_to_ids(toker.tokenize('\n'))[0]
    generation_kwargs = {
        'max_new_tokens': args.max_length,
        'min_length': args.min_length,
        'do_sample': True if (args.top_k > 0 or args.top_p < 1) else False,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'num_beams': args.num_beams,
        'num_return_sequences': args.num_return_sequences,
        'length_penalty': args.length_penalty,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'encoder_no_repeat_ngram_size': args.encoder_no_repeat_ngram_size,
        'pad_token_id': eos_token_id,
        'eos_token_id': eos_token_id,
    }
    if not args.only_evaluate:
        print(json.dumps(generation_kwargs, indent=2, ensure_ascii=False))

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
            #continue
        elif gen_exist and args.only_generate:
            print('gen has existed while metric not required')
            #continue
        elif metric_exist and args.only_evaluate:
            print('metric has existed while gen not required')
            #continue

        metric_res = {}
        collate_fn = getattr(import_module('collators.' + args.collator_name), 'collate_fn')
        if not args.only_generate:
            eval_dataloader = BatchDataLoader(
                data_path=infer_data_path,
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

            _, eval_ppl_micro, eval_acc, eval_rep, eval_wrep, pointwise_loss, pointwise_sample = eval_model_loss(
                accelerator=accelerator,
                model=model,
                eval_dataloader=eval_dataloader,
                epoch_id=0,
                infer=True,
            )
            metric_res['acc'] = float(eval_acc)
            metric_res['rep'] = float(eval_rep)
            metric_res['wrep'] = float(eval_wrep)
            metric_res['ppl_micro'] = float(eval_ppl_micro)
            eval_ppl_list = [np.exp(np.sum(x) / y) for x, y in zip(pointwise_loss, pointwise_sample)]
            eval_ppl_macro = np.mean(eval_ppl_list)
            metric_res['ppl_macro'] = float(eval_ppl_macro)
            assert len(pointwise_loss) == len(pointwise_sample)
            ptr = 0

        if not args.only_evaluate:
            infer_dataloader = BatchDataLoader(
                data_path=infer_data_path,
                batch_size=args.batch_size,
                collate_fn=partial(
                    collate_fn,
                    toker=toker,
                    max_input_length=args.max_input_length,
                    max_decoder_input_length=args.max_decoder_input_length,
                    infer=True,
                ),
                shuffle=False,
            )
            infer_dataloader = accelerator.prepare(infer_dataloader)

            if not args.only_generate:
                from metrics import Metrics
                metrics = Metrics(toker)

            res = []
            decode = lambda x: toker.decode(x, skip_special_tokens=False)
            for batch in tqdm(infer_dataloader, total=len(infer_dataloader), desc='inference', dynamic_ncols=True):
                if 'references' in batch:
                    references = batch.pop('references')
                batch.update(generation_kwargs)
                generations = model.generate(**batch)
                generations = [cut_sequence_to_eos(each, eos_token_id) for each in generations.tolist()]
                batch_size = len(generations) // args.num_return_sequences

                for idx in range(batch_size):
                    if args.num_return_sequences > 1:
                        generation = generations[idx * args.num_return_sequences: (idx+1) * args.num_return_sequences]
                        generation  = [decode(g) for g in generation]
                    else:
                        generation = generations[idx]
                        generation = decode(generation)
                    tmp_res_to_append = {'generation': generation}

                    if not args.only_generate:
                        if args.num_return_sequences == 1:
                            g = generation
                        else:
                            g = generation[0]
                        reference = references[idx]
                        metrics.forword([reference], g, lower=args.lower, chinese=args.chinese)

                        ptr_loss = pointwise_loss[ptr]
                        ptr_sample = pointwise_sample[ptr]
                        turn_loss = sum(ptr_loss)
                        turn_ppl = np.exp(turn_loss / ptr_sample)
                        tmp_res_to_append['token_num'] = int(ptr_sample)
                        tmp_res_to_append['loss'] = turn_loss
                        tmp_res_to_append['ppl'] = turn_ppl
                        ptr += 1

                    res.append(tmp_res_to_append)

            if not args.only_generate:
                assert ptr == len(pointwise_loss)

        if not args.only_evaluate:
            with open(os.path.join(save_path, f'gen.txt'), 'w') as f:
                for line in res:
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')

        metric_res_list = None
        if not args.only_evaluate and not args.only_generate:
            metric_res_list = {}
            closed_res = metrics.close()
            metric_res.update(closed_res[0])
            metric_res_list.update(closed_res[1])

        if not args.only_generate:
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
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_decoder_input_length", type=int, default=48)

    parser.add_argument('--only_evaluate', action='store_true', help='only do evaluation and no inference')
    parser.add_argument('--only_generate', action='store_true', help='do not conduct evaluations')
    parser.add_argument('--chinese', action='store_true')
    parser.add_argument('--lower', action='store_true')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_length", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--encoder_no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    args = parser.parse_args()
    generate(args)


if __name__ == '__main__':
    main()
