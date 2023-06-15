
import logging
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from torch import Tensor

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_model_loss(accelerator, model, eval_dataloader, epoch_id, infer):
    # use the same signature with eval_model_generation
    if accelerator.process_index == 0:
        logger.info('compute eval model loss, using eval mode, '
                    'please change it back to train after calling this function')
    model.eval()
    tot_loss = 0.
    tot_acc = 0.
    tot_rep = 0.
    tot_wrep = 0.
    tot_sample = 0
    pointwise_loss = []
    pointwise_sample = []
    with torch.no_grad():
        if accelerator.process_index == 0:
            pbar = tqdm(eval_dataloader, total=len(eval_dataloader), desc='evaluation', dynamic_ncols=True, leave=True)
        else:
            pbar = eval_dataloader
        for batch in pbar:
            loss_sample, n_sample, acc, rep, wrep, *_ = model(
                validation=True,
                **batch
            )
            if torch.isnan(loss_sample).sum().cpu().long().numpy() > 0:
                logger.info(f'process_index {accelerator.process_index}: NaN occurring!')
                exit()
            tot_loss += loss_sample.sum()
            tot_acc += acc.sum()
            tot_rep += rep.sum()
            tot_wrep += wrep.sum()
            tot_sample += n_sample.sum()
            if infer:
                pointwise_loss.extend(loss_sample.cpu().tolist())
                pointwise_sample.extend(n_sample.cpu().tolist())

    if accelerator.process_index == 0:
        tot_loss = accelerator.reduce(tot_loss)
        tot_sample = accelerator.reduce(tot_sample)

    tot_loss = np.sum(tot_loss.cpu().float().numpy())
    tot_acc = np.sum(tot_acc.cpu().float().numpy())
    tot_rep = np.sum(tot_rep.cpu().float().numpy())
    tot_wrep = np.sum(tot_wrep.cpu().float().numpy())
    tot_sample = np.sum(tot_sample.cpu().float().numpy())
    mean_loss = tot_loss / tot_sample
    mean_ppl = np.exp(mean_loss)
    mean_acc = tot_acc / tot_sample * 100
    mean_rep = tot_rep / tot_sample * 100
    mean_wrep = tot_wrep / tot_sample * 100
    if accelerator.process_index == 0:
        logger.info(f"Epoch {epoch_id}: Val loss {mean_loss} Val ppl {mean_ppl}")
    return mean_loss, mean_ppl, mean_acc, mean_rep, mean_wrep, pointwise_loss, pointwise_sample
