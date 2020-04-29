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


class LSTM(nn.Module):
    def __init__(self,
            vocab_size=None,
            emb_size=None,
            h_size=None,
            padding_idx=None,
            emb_np=None,
            properties=None):
        super(LSTM, self).__init__()

        self.n_properties=len(properties)

        self.word_emb = nn.Embedding(
                vocab_size,
                emb_size,
                padding_idx=padding_idx)
        
        self.word_emb.weight.data.copy_(torch.Tensor(emb_np))

        self.lstm = nn.LSTM(
                input_size=emb_size,    
                hidden_size=h_size,
                num_layers=1,
                bidirectional=False,
                batch_first=True,
                dropout=0.1,
                bias=True)

        #self.mlps = {p: nn.Linear(2 * h_size, 2) for p in PROPERTIES}
        self.mlp = nn.Linear(2*h_size*self.n_properties, self.n_properties)


    def forward(self, sents, sent_lens, preds, heads):

        # Sort the sentences so that the LSTM can process properly
        lens_sorted = sent_lens
        words_sorted = sents
        indices = None
        if(len(sents) > 1):
            lens_sorted, indices = torch.sort(lens_sorted, descending=True)
            words_sorted = words_sorted.index_select(0, indices)

        w_embs = self.word_emb(sents)

        packed_lstm_input = pack_padded_sequence(
                w_embs, lens_sorted, batch_first=True)

        outputs, _ = self.lstm(packed_lstm_input)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Unsort sentences to return to proper alignment with labels
        if len(outputs) > 1:
            outputs = unsort(outputs, indices)

        # outputs : (B, L, h_size), assuming preds (B)
        B, L, h_size = outputs.shape
        pred_reps = outputs[np.arange(B), preds] # expecting (B, h_size)
        head_reps = outputs[np.arange(B), heads] # same as above

        # Get pred-arg representation
        mlp_input = torch.cat([pred_reps, head_reps], dim=-1) # (B, 2*h_size)

        #logits = {p: None for p in PROPERTIES}
        #for p in PROPERTIES:
        #    logits[p] = self.mlps[p](mlp_input)

        mlp_input = mlp_input.repeat(1, self.n_properties) # Don't repeat on axis 0, 
        logits = self.mlp(mlp_input) # (B, properties)

        return logits


        
    def predict(self, sents, sent_lens, preds, heads):

        logits = self.forward(sents, sent_lens, preds, heads)
        preds = ((logits.sign() + 1) / 2).long()
        #preds = F.sigmoid(logits)
        #preds = argmax(logits, -1) # assuming logits a (num_props X 2) array

        return preds
 
def unsort(batch, indices):
    indices_inverted = torch.argsort(indices)
    batch = batch.index_select(0, indices_inverted)
    return batch
