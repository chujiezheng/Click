# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import copy
import datetime
import json
import logging
import os
import math
import shutil
import sys
import time
from tqdm import tqdm
from os.path import join
import numpy as np
from functools import partial
from importlib import import_module

import torch
from torch.optim import AdamW
from transformers.optimization import Adafactor, get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
from accelerate import Accelerator

from utils.building_utils import boolean_string, build_model
from utils.eval_utils import eval_model_loss
from utils.dataloader_utils import BatchDataLoader


def train(args):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)

    init_args_dict = vars(args).copy()

    accelerator = Accelerator()
    if accelerator.process_index == 0:
        logger.info(f'num_processes  :  {accelerator.num_processes}')
        logger.info(f'use_distributed:  {accelerator.use_distributed}')

    set_seed(args.seed)

    assert args.batch_size % args.gradient_accumulation_steps == 0, 'batch size % gradient accumulation steps != 0!'
    args.batch_size = (args.batch_size // args.gradient_accumulation_steps)

    if accelerator.process_index == 0:
        logger.info('train batch size = {}, new train batch size (after gradient accumulation) = {}'.format(
            args.batch_size * args.gradient_accumulation_steps, args.batch_size))

        logger.info('Input Argument Information')
        args_dict = vars(args)
        for a in args_dict:
            logger.info('%-28s  %s' % (a, args_dict[a]))


    #########################################################################
    # Prepare Data Set
    ##########################################################################

    toker = build_model(args, only_toker=True, process_index=accelerator.process_index)
    collate_fn = partial(
        getattr(import_module('collators.' + args.collator_name), 'collate_fn'),
        toker=toker,
        max_input_length=args.max_input_length,
        max_decoder_input_length=args.max_decoder_input_length,
        infer=False,
    )

    train_dataloader = BatchDataLoader(
        data_path=args.train_data_path,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    if args.valid_data_path is not None:
        eval_dataloader = BatchDataLoader(
            data_path=args.valid_data_path,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )
        assert args.valid_step is not None or args.num_epochs is not None
    else:
        eval_dataloader = None


    #########################################################################
    # Prepare Model and Optimizer
    #########################################################################

    _, model = build_model(args, checkpoint=args.load_checkpoint, process_index=accelerator.process_index)

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    if accelerator.process_index == 0:
        logger.info('Number of parameter = {}'.format(total_params))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.adafactor:
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    if args.num_epochs is not None:
        args.num_optim_steps = (args.num_epochs * len(train_dataloader)) // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_optim_steps
    )


    #########################################################################
    # Preparing Training
    ##########################################################################

    if os.path.exists(args.save_path):
        if args.num_epochs is not None:
            checkpoint_name = join(args.save_path, f'epoch-{args.num_epochs-1}.bin')
            if os.path.exists(checkpoint_name):
                exit()
        if accelerator.process_index == 0:
            for file in os.listdir(args.save_path):
                file_path = os.path.join(args.save_path, file)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)

    if accelerator.process_index == 0:
        os.makedirs(args.save_path, exist_ok=True)
        with open(join(args.save_path, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(init_args_dict, f, ensure_ascii=False, indent=2)

        toker.save_pretrained(args.save_path)

        train_logger = open(join(args.save_path, 'train_log.csv'), 'a+', buffering=1)
        eval_logger = open(join(args.save_path, 'eval_log.csv'), 'a+', buffering=1)
        print('epoch,global_step,step,loss,ppl,n_token,epoch_time', file=train_logger)
        print('epoch,global_step,step,loss,ppl', file=eval_logger)

    if accelerator.process_index == 0:
        pbar = tqdm(total=args.num_optim_steps * args.gradient_accumulation_steps, desc=f"training", dynamic_ncols=True)


    #########################################################################
    # Training
    ##########################################################################

    global_step = 0
    step = 0
    epoch = 0
    min_eval_loss = 1e10
    n_token = 0

    train_start_time_epoch = time.time()
    while True:
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.pop('all')
            accelerator.backward(loss)

            ppl = outputs.pop('ppl')
            labels = batch.get('labels', None)

            tmp_n_token = None
            if labels is not None and labels.dim() == 2:
                tmp_n_token = (labels != -100).long().sum()
                if accelerator.num_processes > 1:
                    tmp_n_token = accelerator.reduce(tmp_n_token)
                n_token += int(tmp_n_token.cpu())

            # gradient update
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Print log info to file
                tmp_loss = loss.detach()
                if accelerator.num_processes > 1:
                    tmp_loss = accelerator.reduce(tmp_loss) / accelerator.num_processes
                tmp_ppl = ppl.detach()
                if accelerator.num_processes > 1:
                    tmp_ppl = accelerator.reduce(tmp_ppl) / accelerator.num_processes

                if accelerator.process_index == 0:
                    epoch_time = time.time() - train_start_time_epoch
                    pbar_str = ''#f"tok/s: {n_token_real_all_proc//epoch_time//1000}k "
                    for k, v in outputs.items():
                        pbar_str += f"{k}: {v.item():.2f} "
                    pbar_str += f"loss: {float(tmp_loss.cpu()):.2f} ppl: {float(tmp_ppl.cpu()):.2f} epoch: {epoch}"
                    pbar.set_postfix_str(pbar_str)
                    pbar.update(args.gradient_accumulation_steps)

                    print(f'{epoch},{global_step},{step},{float(tmp_loss.cpu())},{float(tmp_ppl.cpu())},'
                        f'{n_token},{epoch_time}', file=train_logger)

                if args.valid_step is not None and global_step % args.valid_step == 0:
                    if accelerator.process_index == 0:
                        logger.info('current learning rate: ' + str(optimizer.param_groups[0]['lr']))

                    if eval_dataloader is not None:
                        eval_loss, eval_ppl, *_ = eval_model_loss(
                            accelerator=accelerator,
                            model=model,
                            eval_dataloader=eval_dataloader,
                            epoch_id=epoch,
                            infer=False,
                        )
                        if accelerator.process_index == 0:
                            print(f'{epoch},{global_step},{step},{eval_loss},{eval_ppl}', file=eval_logger)

                        if eval_loss < min_eval_loss:
                            min_eval_loss = eval_loss
                            if accelerator.process_index == 0:
                                accelerator.unwrap_model(model).save_pretrained(args.save_path, max_shard_size=args.max_shard_size)
                        model.train()

                if global_step >= args.num_optim_steps:
                    break

        if args.num_epochs is not None:
            if accelerator.process_index == 0:
                logger.info('current learning rate: ' + str(optimizer.param_groups[0]['lr']))

            if eval_dataloader is not None:
                eval_loss, eval_ppl, *_ = eval_model_loss(
                    accelerator=accelerator,
                    model=model,
                    eval_dataloader=eval_dataloader,
                    epoch_id=epoch,
                    infer=False,
                )
                if accelerator.process_index == 0:
                    print(f'{epoch},{global_step},{step},{eval_loss},{eval_ppl}', file=eval_logger)

                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss
                    if accelerator.process_index == 0:
                        accelerator.unwrap_model(model).save_pretrained(args.save_path, max_shard_size=args.max_shard_size)
                model.train()

        epoch += 1
        if global_step >= args.num_optim_steps:
            break

    if accelerator.process_index == 0:
        pbar.close()
        train_logger.close()
        eval_logger.close()

        if eval_dataloader is None:
            accelerator.unwrap_model(model).save_pretrained(args.save_path, max_shard_size=args.max_shard_size)


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

    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--valid_data_path', type=str, default=None)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_decoder_input_length", type=int, default=48)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--adafactor", action='store_true')
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size now means per GPU per step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="to increase effective batch size "
                            "and reduce synchronization")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_optim_steps", type=int, default=None,
                        help="new API specifies num update steps")
    parser.add_argument("--valid_step", type=int, default=None,
                        help="how many optim steps between validations")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="how many training epochs")
    parser.add_argument("--max_shard_size", type=str, default='5GB')

    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

    args = parser.parse_args()
    assert args.valid_step is None or args.num_epochs is None
    train(args)


if __name__ == '__main__':
    main()
