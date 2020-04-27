import logging
import random

import numpy as np
import torch
from torch import unsqueeze
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, \
        pack_padded_sequence, pad_sequence
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


class LogReg():
    def __init__(
            self,
            random_state=None,
            solver='lbfgs',
            penalty='l2',
            properties=None):

        self.clfs = {}
        for p in properties:
            self.clfs[p] = LogisticRegression(
                    random_state=random_state,
                    solver=solver,
                    penalty=penalty,
                    max_iter=1000)


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

