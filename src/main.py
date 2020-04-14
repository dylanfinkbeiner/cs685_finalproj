import pickle
from tqdm import tqdm
from collections import defaultdict
import os
import random

import pandas as pd
from nltk.corpus import ptb
from nltk.tree import ParentedTree
from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from arg_parser import get_args
import models
import data_utils
import evaluate
import train

from names import *

# Want to figure out a way to define this once across all files
#SPLITS = ['train', 'dev', 'test'] 
#
#DATA_DIR = '../data/'
#PICKLED_DIR = os.path.join(DATA_DIR, 'pickled')
#MODEL_DIR = '../saved_models/'
#PROTO_TSV = '../protoroles_eng_pb_08302015.tsv'
#
#PROPERTIES = ['instigation', 'volition', 'awareness', 'sentient',
#'exists_as_physical', 'existed_before', 'existed_during', 'existed_after',
#'created', 'destroyed', 'predicate_changed_argument', 'change_of_state', 
#'changes_possession', 'change_of_location', 'stationary', 'location_of_event', 
#'makes_physical_contact', 'manipulated_by_another']

def get_data(args):
    df = pd.read_csv(PROTO_TSV, sep='\t')

    # Sentences
    sent_ids = set(df['Sentence.ID'].tolist())
    print(f'There are {len(sent_ids)} unique sentences.')
    path = os.path.join(DATA_DIR, 'sents.pkl')
    sents_data = None
    if os.path.exists(path) and not args.init_sents:
        with open(path, 'rb') as f:
            sents_data = pickle.load(f)
    else:
        with open(path, 'wb') as f:
            sents_data = data_utils.get_nltk_sents(sent_ids)
            pickle.dump(sents_data, f)

    # Instances
    path = os.path.join(DATA_DIR, 'instances.pkl')
    proto_instances = None
    possible = None # Data to compare to SPRL paper
    if os.path.exists(path) and not args.init_instances:
        with open(path, 'rb') as f:
            proto_instances, possible = pickle.load(f)
    else:
        with open(path, 'wb') as f:
            proto_instances, possible = data_utils.build_instance_list(df)
            data_utils.add_pred_args(proto_instances, sents_data['trees'])
            pickle.dump((proto_instances, possible), f)

    num_instances = sum([len(x) for x in proto_instances.values()])
    print(f'There are {num_instances} instances.')

    return df, proto_instances, possible, sents_data


if __name__ == '__main__':
    args = get_args()

    # Things that do not change depending on experiment
    df, proto_instances, possible, sents = get_data(args)

    # Now normalize (which might be different per experiment)
    #data_utils.normalize(proto_instances, args)

    # TODO Why are these numbers not matching the ones in SPRL paper?
    for p in PROPERTIES:
        possible_train = possible['train'][p]
        possible_dev = possible['dev'][p]
        print(f'{p} -- Train: {possible_train}, Dev: {possible_dev}\n')


    X = {}
    y = {}
    for split in SPLITS:
        X_split, y_split = data_utils.get_ins_outs(args, proto_instances[split],
                properties=PROPERTIES, sents=sents)
        X[split] = X_split
        y[split] = y_split
    

    # Setting up models and training, evaluating
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #presence = vectorized > 0

    ##X, y = vectorized, labels
    #X, y = presence, labels

    model = None
    if args.model_type == 'majority':
        model_path = os.path.join(MODEL_DIR, 'majority.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = models.MajorityBaseline(proto_instances, PROPERTIES)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    elif args.model_type == 'logreg':
        data_utils.unkify(X, cutoff=2)

        all_X = []
        l_u = {}
        l = 0
        u = 0
        for split in SPLITS:
            all_X.extend(X[split])
            u += len(X[split])
            l_u[split] = (l, u)
            l = u

        v = DictVectorizer(sparse=False)
        vectorized = v.fit_transform(all_X)
        names = np.array(v.get_feature_names())

        for split in SPLITS:
            l, u = l_u[split]
            X[split] = vectorized[l:u]

        # Split up by proto role property
        for split in SPLITS:
            y_p = data_utils.split_on_properties(y[split], PROPERTIES)
            y[split] = y_p

        model = models.LogReg(
                random_state=args.seed, 
                solver='lbfgs', 
                penalty='l2',
                properties=PROPERTIES)

    metrics = train.train(args, model, X, y, PROPERTIES)

    for k, v in metrics.items():
        print(f"{k}: F={v['F']:.2f}, p={v['precision']:.2f}, r={v['recall']:.2f}")

    print('Finished training.')
    breakpoint()

    results = evaluate.evaluate(args, model, X['test'], y['test'])

    print('End of main.')
    breakpoint()


