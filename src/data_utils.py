from collections import defaultdict
from tqdm import tqdm
from random import shuffle
from nltk.corpus import ptb
from nltk.tree import ParentedTree
from typing import Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import numpy as np


SPLITS = ['train', 'dev', 'test'] 


def get_pred_arg(pt, ptree):
    pred = get_predicate(pt['Pred.Token'], ptree)

    arg = []
    arg_idx = []
    arg_height_pairs = [t.split(':') for t in pt['Arg.Pos'].split(',')]
    # Treats all cases, including discontiguous arguments
    for pair in arg_height_pairs:
        arg_id, height = [int(n) for n in arg_height_pairs[0]]
        arg_ , arg_idx_ = get_argument(arg_id, height, ptree)
        arg.extend(arg_)
        arg_idx.extend(arg_idx_)

    return pred, arg, arg_idx


def get_predicate(terminal_id, tree):
    return tree.leaves()[terminal_id]


def get_argument(terminal_id, height, ptree):
    terminals = get_terminals(ptree)

    parent = terminals[terminal_id]
    if height > 0:
        for _ in range(height):
            parent = parent.parent()

    arg_leaves = parent.leaves()
    sent_leaves = ptree.leaves()

    # Seems like a clunky way to get indices
    arg_idx = []
    j = 0
    for i, leaf in enumerate(sent_leaves):
        if leaf == arg_leaves[j]:
            arg_idx.append(i)
            j += 1
        if j >= len(arg_leaves):
            break

    return arg_leaves, arg_idx


# NOTE Kind of a heuristic, seems to work fine though
def get_terminals(ptree: ParentedTree) -> list:
    terms = ptree.subtrees(filter=lambda x: len(list(x.subtrees())) == 1)
    terms = list(terms)
    assert len(ptree.leaves()) == len(terms) # Pull out to unit test?

    return terms


# Properties is just a list of proto-properties, for ease of work
def get_ins_outs(args, data_points, properties=None, sents=None) -> Tuple[list,
        list]:
    possible_feats = ['pred_lemma_emb', 'pred_pos', 'arg_direction']

    X = []
    y = []

    if args.model_type == 'majority':
        for pt in data_points:
            for p in properties:
                X.append(p)
                y.append(pt[p]['binary'])
    else:
        for pt in data_points:
            # One training example per property
            for p in properties:
                X_d = {}

                #for feat in args.features:
                    #add_feature(feat, X_d, pt, sents)
                
                if 'pred_lemma' in args.features:
                    X_d['pred_lemma'] = pt['pred_lemma']
                if 'pred_lemma_emb' in args.features:
                    pass
                if 'arg_direction' in args.features:
                    if pt['Pred.Token'] < pt['arg_indices'][0]:
                        X_d['arg_direction'] = 1
                    elif pt['Pred.Token'] > pt['arg_indices'][0]:
                        X_d['arg_direction'] = -1
                    else:
                        raise Exception
                if 'arg_distance' in args.features:
                    arg_idx = pt['arg_indices']
                    arg_mid = (arg_idx[0] + arg_idx[-1]) / 2
                    X_d['arg_distance'] = abs(arg_mid - float(pt['Pred.Token']))


                X.append((p, X_d)) # (property, feats)
                y.append(pt[p]['binary'])

    return X, y


def add_pred_args(proto_instances, trees):

    normalized = {}
    # First, get the predicate and arg tokens
    for split in SPLITS:
        data_points = proto_instances[split]
        for pt in data_points:
            s_id = pt['Sentence.ID']
            tree = trees[s_id]
            ptree = ParentedTree.convert(tree)
            pred, arg, arg_idx = get_pred_arg(pt, ptree)

            pt['pred_token'] = pred
            pt['arg_tokens'] = arg
            pt['arg_indices'] = arg_idx

    return


def normalize(proto_instances, args):
    # TODO
    return


def top_ten_logreg(clf, names):
    coef = clf.coef_.flatten()

    # Top ten for impersonal (0-label)
    idx = np.argsort(coef)
    print(f'Top 10 personal: {names[idx][-10:]}')
    print(f'Top 10 impersonal: {names[idx][:10]}')
    

def get_nltk_sents(sent_ids):
    raw = {}
    tagged = {}
    trees = {}
    data = {}

    for i in tqdm(sent_ids, desc="Collecting sentences and trees"):
        file_num, sent_num = i.split('_')
        subdir = file_num[:2]
        sent_num = int(sent_num)
        path = f'WSJ/{subdir}/WSJ_{file_num}.MRG'
        raw[i] = ptb.sents(path)[sent_num]
        tagged[i] = ptb.tagged_sents(path)[sent_num]
        trees[i] = ptb.parsed_sents(path)[sent_num]

    data['raw'] = raw
    data['tagged'] = tagged
    data['trees'] = trees

    return data


def build_instance_list(df):
    print(f'df has {df.shape[0]} entries')

    instances = {name: dict() for name in SPLITS}

    props = ['Response', 'Applicable']
    cols = set(df.columns.tolist())
    not_props = cols - set(props) - {'Property'}

    properties = list(set(df['Property'].tolist()))
    possible = {split:{p:0 for p in properties} for split in ['train', 'dev']}

    for _, x in tqdm(df.iterrows(), desc='Processing proto-role data entries:'):
        idstr = [x['Sentence.ID'], x['Pred.Token'], x['Arg'], x['Arg.Pos']]
        idstr = '_'.join([str(n) for n in idstr])

        # First time for this unique instance: populate
        if idstr not in instances[x['Split']]:
            d = {}
            for col in not_props:
                d[col] = x[col]

            # Lemma contained in PropBank roleset
            d['pred_lemma'] = x['Roleset'].split('.')[0]

            instances[x['Split']][idstr] = d

        # Not first time: just retrieve
        else:
            d = instances[x['Split']][idstr]

        d[x['Property']] = {p: x[p] for p in props}
        binary_label = int(x['Applicable'] and (x['Response'] >= 4))
        d[x['Property']]['binary'] = binary_label
        
        if binary_label == 1 and x['Split'] != 'test':
            possible[x['Split']][x['Property']] += 1

    # ID keys in instances serve no further purpose, listify
    for split in SPLITS:
        data_points = instances[split]
        listified = sorted(data_points.items(), key=lambda x: x[0])
        instances[split] = [x[1] for x in listified]

    return instances, possible

