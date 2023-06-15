# coding=utf-8

import json
import warnings
import numpy as np
import nltk
import copy
from typing import List
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(0, len(sub)+1)] for _ in range(0, len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1, len(string) + 1):
            if string[i-1] == sub[j-1]:
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j], lengths[i][j-1])

    return lengths[len(string)][len(sub)]


class Metrics(object):
    def __init__(self, toker=None):
        self.refs = []
        self.hyps = []
        self.toker = toker

    def forword(self, refs: str, hyp: str, lower=False, chinese=False): # TODO: only applicable to English
        if not chinese:
            self.refs.append([nltk.word_tokenize(e.lower() if lower else e) for e in refs])
            self.hyps.append(nltk.word_tokenize(hyp.lower() if lower else hyp))
        else:
            self.refs.append([self.toker.tokenize(e) for e in refs])
            self.hyps.append(self.toker.tokenize(hyp))

    def set_refs(self, refs):
        self.refs = copy.deepcopy(refs)

    def set_hyps(self, hyps):
        self.hyps = copy.deepcopy(hyps)

    def calculate_bleu_k(self, k):
        weights = [1. / k] * k + (4 - k) * [0.]
        try:
            bleu = corpus_bleu(self.refs, self.hyps, weights=weights,
                               smoothing_function=SmoothingFunction().method3)
        except ZeroDivisionError as _:
            warnings.warn('the bleu is invalid')
            bleu = 0.
        return bleu

    def calculate_distinct_k(self, k):
        ngrams = []
        for sen in self.hyps:
            tmp_ngrams = list(zip(*[sen[i:] for i in range(k)]))
            ngrams.extend(tmp_ngrams)
        ngrams = Counter(ngrams)
        dist = len(ngrams) / max(sum(ngrams.values()), 1e-10)
        return dist

    """
    def calculate_repetition_k(self, k):
        repetitions = []
        for sen in self.hyps:
            tmp_ngrams = list(zip(*[sen[i:] for i in range(k)]))
            tmp_ngrams = Counter(tmp_ngrams)
            tmp_dist = len(tmp_ngrams) / max(sum(tmp_ngrams.values()), 1e-10)
            repetitions.append(1. - tmp_dist)
        return np.mean(repetitions), repetitions
    """

    def calculate_repetition_k(self, k):
        count = [0, 0]
        for sen in self.hyps:
            tmp_ngrams = list(zip(*[sen[i:] for i in range(k)]))
            count[1] += len(tmp_ngrams)
            tmp_ngrams = Counter(tmp_ngrams)
            count[0] += len(tmp_ngrams)
        return 1. - count[0] / count[1]

    def calculate_unigram_f1(self):
        f1_scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            scores = []
            for ref in refs:
                cross = Counter(hyp) & Counter(ref)
                cross = sum(cross.values())
                p = cross / max(len(hyp), 1e-10)
                r = cross / max(len(ref), 1e-10)
                f1 = 2 * p * r / max(p + r, 1e-10)
                scores.append(f1)
            f1_scores.append(max(scores))
        return np.mean(f1_scores), f1_scores

    def calculate_rouge_k(self, k):
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            rec = []
            hyp_kgrams = Counter(zip(*(hyp[i:] for i in range(k))))
            for ref in refs:
                ref_kgrams = Counter(zip(*(ref[i:] for i in range(k))))
                cross_kgrams = hyp_kgrams & ref_kgrams
                rec.append(sum(cross_kgrams.values()) / max(sum(ref_kgrams.values()), 1e-10))
            score = max(rec)
            scores.append(score)
        return np.mean(scores), scores

    def calculate_rouge_l(self, beta=1.2):
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            prec = []
            rec = []
            for ref in refs:
                lcs = my_lcs(ref, hyp)
                prec.append(lcs / max(len(hyp), 1e-10))
                rec.append(lcs / max(len(ref), 1e-10))
            prec_max = max(prec)
            rec_max = max(rec)
            if prec_max != 0 and rec_max !=0:
                score = ((1 + beta**2) * prec_max * rec_max)/float(rec_max + beta**2 * prec_max)
            else:
                score = 0.0
            scores.append(score)
        return np.mean(scores), scores

    def close(self):
        result = {}
        result_list = {}

        result['length'] = np.mean(list(map(len, self.hyps)))

        for k in range(1, 5):
            bleu = self.calculate_bleu_k(k)
            result[f'bleu-{k}'] = 100 * bleu

        for k in range(1, 4):
            dist = self.calculate_distinct_k(k)
            result[f'dist-{k}'] = 100 * dist

        for k in range(1, 3):
            rouge, scores = self.calculate_rouge_k(k)
            result[f'rouge-{k}'] = 100 * rouge
            result_list[f'rouge-{k}'] = scores

        for k in range(2, 5):
            rep = self.calculate_repetition_k(k)
            result[f'rep-{k}'] = 100 * rep
        result['diversity'] = (1. - result['rep-2'] / 100) * (1. - result['rep-3'] / 100) * (1. - result['rep-4'] / 100)

        f1, scores = self.calculate_unigram_f1()
        result['f1'] = 100 * f1
        result_list['f1-l'] = scores

        rl, scores = self.calculate_rouge_l()
        result['rouge-l'] = 100 * rl
        result_list['rouge-l'] = scores

        return result, result_list
