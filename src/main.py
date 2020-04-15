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


def get_data(args):
    df = pd.read_csv(PROTO_TSV, sep='\t')

    # Sentences
    sent_ids = set(df['Sentence.ID'].tolist())
    print(f'There are {len(sent_ids)} unique sentences.')
    path = os.path.join(PICKLED_DIR, 'sents.pkl')
    sents_data = None
    if os.path.exists(path) and not args.init_sents:
        with open(path, 'rb') as f:
            sents_data = pickle.load(f)
    else:
        with open(path, 'wb') as f:
            sents_data = data_utils.get_nltk_sents(sent_ids)
            pickle.dump(sents_data, f)

    path = os.path.join(PICKLED_DIR, 'dependencies.pkl')
    if os.path.exists(path) and not args.init_deps:
        with open(path, 'rb') as f:
            dependencies = pickle.load(f)
    else:
        with open(path, 'wb') as f:
            #sent_ids = list(sents_data['raw'].keys())
            dependencies = data_utils.get_dependencies(sent_ids)
            pickle.dump(dependencies, f)
    sents_data['dependencies'] = dependencies

    data_utils.match_to_raw(sents_data['raw'], dependencies)

    #for sent_id in sent_ids:
    #    try:
    #        raw = sents_data['raw'][sent_id]
    #        dep = dependencies[sent_id]
    #        assert len(raw) == len(dep)
    #    except Exception:
    #        breakpoint()

    # Instances
    path = os.path.join(PICKLED_DIR, 'instances.pkl')
    proto_instances = None
    possible = None # Data to compare to SPRL paper
    if os.path.exists(path) and not args.init_instances:
        with open(path, 'rb') as f:
            proto_instances, possible = pickle.load(f)
    else:
        proto_instances, possible = data_utils.build_instance_list(df)
        with open(path, 'wb') as f:
            pickle.dump((proto_instances, possible), f)
    if args.add_pred_args:
        data_utils.add_pred_args(proto_instances, sents_data['trees'])
        with open(path, 'wb') as f:
            pickle.dump((proto_instances, possible), f)

    w2e = None
    path = os.path.join(PICKLED_DIR, 'glove.pkl')
    if os.path.exists(path) and not args.init_glove:
        with open(path, 'rb') as f:
            w2e = pickle.load(f)
    else:
        w2e = data_utils.w2e_from_file(GLOVE_FILE)
        with open(path, 'wb') as f:
            pickle.dump(w2e, f)


    # Little test to make sure arg_indices make sense
    #for split in SPLITS:
    #    for pt in proto_instances[split]:
    #        pred_idx = pt['Pred.Token']
    #        first_arg = pt['arg_indices'][0]
    #        last_arg = pt['arg_indices'][-1]
    #        if first_arg < pred_idx:
    #            assert last_arg < pred_idx
    #        elif first_arg > pred_idx:
    #            assert last_arg > pred_idx
    #        else: # Arg index NEVER equal pred index
    #            raise Exception


    num_instances = sum([len(x) for x in proto_instances.values()])
    print(f'There are {num_instances} instances.')

    return df, proto_instances, possible, sents_data, w2e


if __name__ == '__main__':
    args = get_args()

    # Things that do not change depending on experiment
    df, proto_instances, possible, sents, w2e = get_data(args)

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
                properties=PROPERTIES, sents=sents, w2e=w2e)
        X[split] = X_split
        y[split] = y_split

    print('Got features.')

    # Setting up models and training, evaluating
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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


