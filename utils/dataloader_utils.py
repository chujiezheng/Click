import json
import os
import math
from functools import partial
import nltk
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler


def _norm(s):
    return ' '.join(s.strip().split())


def _norm1(x):
    x = " ".join(x.strip().split())
    xs = []
    for e in nltk.sent_tokenize(x):
        xs.append(e.capitalize())
    x = ' '.join(xs)
    return x


def _norm2(x: str):
    return " ".join(x.strip().split()).replace(" l o l ", " lol ").replace(" ' m ", "'m ")\
        .replace(" ’ t ", "'t ").replace(" ’ ll ", "'ll ").replace(" ’ s ", "'s ").replace(" ’ ve ", "'ve ").replace(" ’ re ", "'re ")\
        .replace(" ' t ", "'t ").replace(" ' ll ", "'ll ").replace(" ' s ", "'s ").replace(" ' ve ", "'ve ").replace(" ' re ", "'re ")\
        .replace(" .", ".").replace(" ,", ",").replace(" ?", '?').replace(" !", '!')


class BasicDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, i):
        return self.data_list[i]

    def __len__(self):
        return len(self.data_list)


class BatchDataLoader(DataLoader):
    def __init__(self,
            data_list=None, data_path=None, batch_size=None,
            collate_fn=None, shuffle=True, num_workers=16,
        ):
        if data_list is None:
            data_list = [json.loads(e) for e in open(data_path)]
        dataset = BasicDataset(data_list)
        basic_sampler = RandomSampler if shuffle else SequentialSampler
        sampler = BatchSampler(basic_sampler(dataset), batch_size=batch_size, drop_last=False)
        super().__init__(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)
