import logging
import random

import numpy as np
import torch
from torch import unsqueeze
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, \
        pack_padded_sequence, pad_sequence


class LogReg(nn.Module):
    def __init__(self):
        super(LogReg, self).__init__()

        self.word_emb = nn.Embedding(
                vocab_size,
                emb_size,
                padding_idx=padding_idx)


    def forward(self, inputs):
        w_embs = self.word_emb(inputs) # (B, L, D)

        avg = w_embs.mean(1) # (B, D)

        logits = self.logreg(avg) # (B, 2)

        return logits



class Glove_LogReg(nn.Module):
    def __init__(
            self,
            vocab_size=None,
            emb_size=None,
            padding_idx=None,
            emb_np=None):
        super(Glove_LogReg, self).__init__()

        self.word_emb = nn.Embedding(
                vocab_size,
                emb_size,
                padding_idx=padding_idx)
        #self.word_emb.weight.requires_grad = False

        # Copy word embeddings from numpy array
        self.word_emb.weight.data.copy_(torch.Tensor(emb_np))

        # Takes avg embedding to 2 output classes
        self.logreg = nn.Linear(emb_size, 2)


    def forward(self, inputs):
        w_embs = self.word_emb(inputs) # (B, L, D)

        avg = w_embs.mean(1) # (B, D)

        logits = self.logreg(avg) # (B, 2)

        return logits



class MajorityBaseline():
    def __init__(self, instances, properties):
        # Get preferred output for each proto-role property
        counts = {p:0 for p in properties}

        total_instances = 0
        double_check = {0:0, 1:0}

        for split_name in ['train', 'dev']:
            split = instances[split_name]
            total_instances += len(split)
            for d in split.values():
                for p in properties:
                    label = d[p]['binary']
                    counts[p] += label
                    double_check[label] += 1

        assert sum(double_check.values()) == (total_instances * len(properties))

        self.prefs = {}
        for p in properties:
            self.prefs[p] = int(counts[p] > (total_instances // 2))

        print('Finished Majority init')
        breakpoint()


    # NOTE interesting question for later: how could this code be vectorized?
    def predict(self, inputs):
        preds = np.zeros(len(inputs))
        for i, x in enumerate(inputs):
            preds[i] = self.prefs[x]

        return preds

