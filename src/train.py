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

GLOVE_TXT =  './glove.twitter.27B.100d.txt'
GLOVE_PKL =  './glove.pkl'
WEIGHTS_DIR = './weights'


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




# Turn normalized tweets into dicts containing tokens and counts
def countify(normalized):
    counts = []
    for tweet in normalized:
        d = defaultdict(int)
        for token in tweet:
            d[token] += 1
        
        counts.append(dict(d))
    return counts


def build_vocabs(tokenized):
    tokens = [w.lower() for tweet in tokenized for w in tweet]
    types = set(tokens)

    w2i = defaultdict(lambda: len(w2i))
    i2w = dict()

    i2w[w2i['<pad>']] = '<pad>'
    i2w[w2i['<unk>']] = '<unk>'

    for t in types:
        i2w[w2i[t]] = t

    return dict(w2i), i2w


def numericalize(normalized, w2i):
    # normalized should be a LIST of lists of strings

    numericalized = []
    for tweet in normalized:
        a = np.zeros(len(tweet))
        for i, token in enumerate(tweet):
            a[i] = w2i[token]
        numericalized.append(a)

    return numericalized


def unkify(normalized, new_w2i=None):
    #emojis = list(emoji.UNICODE_EMOJI)
    for tweet in normalized:
        for i, token in enumerate(tweet):
            #if (token not in unk_dict 
            #        or unk_dict[token]) and token not in emojis:
            if token not in new_w2i:
                tweet[i] = '<unk>'


def train(args, data=None):
    if args.model_type == 'logreg':
                clf_overall = LogisticRegression(
                random_state=args.seed, solver='lbfgs', penalty='l2').fit(X, y)

    #elif args.model_type == 'glove':
    else:
        # Round up data
        w2i, i2w = build_vocabs(normalized)
        glove_data = fetch_glove_data(args, w2i=w2i, i2w=i2w)
        w2i = glove_data['w2i']
        i2w = glove_data['i2w']
        unkify(normalized, new_w2i=w2i)
        numericalized = numericalize(normalized, w2i)

        # Initialize model 
        if args.model_type == 'glove':
            print('Using Glove LogReg model') 
            clf = models.Glove_LogReg(
                vocab_size=len(glove_data['word_list']),
                emb_size=(glove_data['emb_np']).shape[1],
                padding_idx=0,
                emb_np=glove_data['emb_np'])

        elif args.model_type == 'better':
            print('Using better model') 
            clf = models.DAN(
                vocab_size=len(w2i), 
                emb_size=(glove_data['emb_np']).shape[1],
                padding_idx=0,
                emb_np=glove_data['emb_np'],
                h_size=args.h_size)


        weights_path = os.path.join(WEIGHTS_DIR, args.model)
        i = 0
        attempt = weights_path + f'_{i:02d}'
        while os.path.exists(attempt):
            i += 1
            attempt = weights_path + f'_{i:02d}'
        weights_path = attempt
        torch.save(clf.state_dict(), weights_path) # Save random reset

        # Needed when data in list form
        def train_test(data, train_idx, test_idx):
            train = [data[i] for i in train_idx]
            test = [data[i] for i in test_idx] 
            return train, test

        # Training loop
        X, y = numericalized, labels
        try:
            n_correct, tpp, fpp, fnp, tpi, fpi, fni= 0,0,0,0,0,0,0
            for k, (train, test) in enumerate(kf.split(X)):
                # Reset to random, otherwise K-Fold makes no sense
                clf.load_state_dict(torch.load(weights_path))
                opt = Adam(clf.parameters(), lr=args.lr, betas=[0.9, 0.9])
                #opt = AdamW(clf.parameters(), lr=args.lr, betas=[0.9, 0.9])
                #opt = SGD(clf.parameters(), lr=args.lr, momentum=0.9,
                #        nesterov=True)

                X_train, X_test = train_test(X, train, test)
                y_train, y_test = train_test(y, train, test)
                n_train_batches = (len(train) // args.batch_size) + 1

                # Batch loader for this fold's train data
                batch_loader = data_utils.batch_loader(
                    list(zip(X_train, y_train)), batch_size=args.batch_size, shuffle_idx=True)

                for e in range(args.epochs):
                    clf.train()
                    #for b in tqdm(
                    #        range(n_train_batches), 
                    #        ascii=True, desc=f'Epoch {e+1}/{args.epochs} progress', ncols=80):
                    for b in range(n_train_batches):

                        opt.zero_grad()
                        batch = next(batch_loader)
                        if args.model_type == 'glove':
                            logits = clf(batch['tweets'])
                        else:
                            logits = clf(batch['tweets'], batch['sent_lens'])

                        loss = F.cross_entropy(logits, batch['labels'])
                        #print(f'Loss: {loss.item()}')
                        loss.backward()
                        opt.step()


                if args.model_type == 'glove':
                    top_ten_glove(clf, glove_data['word_list'])

                with torch.no_grad():
                    n_correct_, tpp_, fpp_, fnp_, tpi_, fpi_, fni_ = evaluate(
                            args, clf, X_test, y_test, i2w=i2w)
                n_correct += n_correct_
                tpp += tpp_
                fpp += fpp_
                fnp += fnp_
                tpi += tpi_
                fpi += fpi_
                fni += fni_
            

            acc = n_correct / 250
            precision_p = tpp / (tpp + fpp)
            recall_p = tpp / (tpp + fnp)
            precision_i = tpi / (tpi + fpi)
            recall_i = tpi / (tpi + fni)
            print(f'Accuracy {acc}\nPrecision p {precision_p}\nRecall p {recall_p}')
            print(f'\nPrecision i {precision_i}\nRecall i {recall_i}')
            

        except KeyboardInterrupt:
            inp = input('Training interrupted: Save weights? [y/n]')
            if inp.lower().strip() == 'y':
                #torch.save(clf.state_dict(), weights_path)
                pass

        finally:
            print(f'Training complete for model: {args.model}')


def unnumericalize(array, i2w):
    s = [i2w[x] for x in array]
    return ' '.join(s)

