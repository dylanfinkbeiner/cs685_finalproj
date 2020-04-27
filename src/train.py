from collections import defaultdict
import os
import pickle
from tqdm import tqdm
import random

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
import emoji


import models
import data_utils
import evaluate

#TODO Currently does not return !!!
def top_ten_logreg(clf, names):
    coef = clf.coef_.flatten()

    # Top ten for impersonal (0-label)
    idx = np.argsort(coef)
    print(f'Top 10 personal: {names[idx][-10:]}')
    print(f'Top 10 impersonal: {names[idx][:10]}')


def top_ten_glove(clf, names):
    names = np.array(names)

    emb = clf.word_emb.weight # (V, D)
    weights = clf.logreg.weight # (2, D)
    mm = torch.mm(weights, emb.T) # (2, V)
    idx = torch.argsort(mm, dim=1) # (2, V)

    print(f"Top 10 'i': {names[idx[0]][-10:]}")
    print(f"Top 10 'p': {names[idx[1]][-10:]}")


def train(args, model, X, y, properties):
    X_train, y_train = X['train'], y['train']
    X_dev, y_dev = X['dev'], y['dev']

    if args.model_type == 'logreg':

        results = {p: None for p in properties}
        metrics = {p: None for p in properties}
        train_metrics = {p: None for p in properties}
        for p in tqdm(properties, desc=f'Training clf'):
            print(f'...for prop {p}')
            clf_p = model.clfs[p]

            y_p = y_train[p]

            clf_p.fit(X['train'], y_p)

            _, train_metrics[p] = evaluate.evaluate(args, clf_p, X['train'],
                    y_p)

            results[p], metrics[p] = evaluate.evaluate(args, clf_p, X['dev'],
                    y['dev'][p])

        #metric_names = ['F', 'precision', 'recall']


    if args.model_type == 'majority':
        results = {}
        metrics = {}
        results['all'], metrics['all'] = evaluate.evaluate(args, model, X['dev'], y['dev'])

    breakpoint()
    F, precision, recall = evaluate.micro_average(results)
    metrics['micro_avg'] = {'F': F, 'precision': precision, 'recall': recall}

    return metrics
