from collections import defaultdict
import os
import pickle
from tqdm import tqdm
import random
import math

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW

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


def train(args, model, X, y):
    X_train, y_train = X['train'], y['train']
    X_dev, y_dev = X['dev'], y['dev']

    if args.model_type == 'lstm':
        train_lstm(args, model, X, y)

    elif args.model_type == 'logreg':

        results = {p: None for p in PROPERTIES}
        metrics = {p: None for p in PROPERTIES}
        train_metrics = {p: None for p in PROPERTIES}
        for p in tqdm(PROPERTIES, desc=f'Training clf'):
            print(f'...for prop {p}')
            clf_p = model.clfs[p]

            y_p = y_train[p]

            clf_p.fit(X['train'], y_p)

            _, train_metrics[p] = evaluate.evaluate(args, clf_p, X['train'],
                    y_p)

            results[p], metrics[p] = evaluate.evaluate(args, clf_p, X['dev'],
                    y['dev'][p])

        #metric_names = ['F', 'precision', 'recall']


    elif args.model_type == 'majority':
        results = {}
        metrics = {}
        results['all'], metrics['all'] = evaluate.evaluate(args, model, X['dev'], y['dev'])

    breakpoint()
    F, precision, recall = evaluate.micro_average(results)
    metrics['micro_avg'] = {'F': F, 'precision': precision, 'recall': recall}

    return metrics


def train_lstm(args, model, X, y):


    # Data loaders
    loader_train = data_loader(X['train'], y['train'],
            batch_size=args.batch_size, shuffle_idx=True)
    n_train_batches = math.ceil(len(X['train']) / args.batch_size)

    # Optimizer
    opt = Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.9])

    # Train loop
    try:
        for e in range(args.epochs):
            for b in tqdm(
                    range(n_train_batches), 
                    ascii=True, 
                    desc=f'Epoch {e+1}/{args.epochs} progress', 
                    ncols=80):
                opt.zero_grad()
                sents, sent_lens, preds, heads, labels = next(loader_train)
                logits = model(sents, sent_lens, preds, heads)
                loss = bce_loss(logits, labels)
                loss.backward()
                opt.step()

    except KeyboardInterrupt:
        pass
    # End of train loop
    return


def bce_loss(logits, labels):
    # Expected labels : (B, num_properties)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss


def data_loader(X, y, batch_size=None, shuffle_idx=False):
    data = list(zip(X, y))
    idx = list(range(len(data)))
    while True:
        if shuffle_idx:
            random.shuffle(idx) # In-place shuffle
        
        for span in idx_spans(idx, batch_size):
            batch = [data[i] for i in span]
            yield prepare_batch(batch)


def idx_spans(idx, span_size):
    for i in range(0, len(idx), span_size):
        yield idx[i:i+span_size]


def prepare_batch(batch):
    # batch[i] = X, y
    batch_size = len(batch)
    sent_lens = torch.LongTensor([len(x[0][0]) for x in batch])
    max_length = torch.max(sent_lens).item()
    n_properties = len(batch[0][1])

    # Zero is padding index
    sents = torch.zeros((batch_size, max_length)).long()
    preds = torch.zeros(batch_size).long()
    heads = torch.zeros(batch_size).long()
    labels = torch.zeros(batch_size, n_properties)

    for i, (X_batch, y_batch) in enumerate(batch):
        sent, (pred_idx, head_idx) = X_batch
        sents[i,:len(sent)] = torch.LongTensor(sent)
        preds[i] = pred_idx
        heads[i] = head_idx
        labels[i] = torch.tensor(y_batch)

    return sents, sent_lens, preds, heads, labels
