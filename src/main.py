import pickle
from tqdm import tqdm
from collections import defaultdict
import os

import pandas as pd
from nltk.corpus import ptb
from nltk.tree import ParentedTree
from matplotlib import pyplot as plt

from argparser import get_args
from models import MajorityBaseline
import data_utils
import eval

DATA_DIR = '../pickled_data/'
MODEL_DIR = '../saved_models/'
proto_tsv = '../protoroles_eng_pb_08302015.tsv'


def fetch_nltk_data(sent_ids):
    sents = {}
    tagged_sents = {}
    trees = {}
    data = {}

    for i in tqdm(sent_ids, desc="Collecting sentences and trees"):
        fnum, snum = i.split('_')
        d = fnum[:2]
        snum = int(snum)
        path = f'WSJ/{d}/WSJ_{fnum}.MRG'
        sents[i] = ptb.sents(path)[snum]
        tagged_sents[i] = ptb.tagged_sents(path)[snum]
        trees[i] = ptb.parsed_sents(path)[snum]

    data['sents'] = sents
    data['tagged_sents'] = tagged_sents
    data['trees'] = trees

    return data


def build_instance_list(df):
    print(f'df has {df.shape[0]} entries')

    splits = ['train', 'dev', 'test']
    instances = {name: dict() for name in splits}

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
            instances[x['Split']][idstr] = d

        # Not first time: just retrieve
        else:
            d = instances[x['Split']][idstr]

        d[x['Property']] = {p: x[p] for p in props}
        binary_label = int(x['Applicable'] and (x['Response'] >= 4))
        d[x['Property']]['binary'] = binary_label
        
        if binary_label == 1 and x['Split'] != 'test':
            possible[x['Split']][x['Property']] += 1


    return instances, possible


def get_data(args):
    df = pd.read_csv(proto_tsv, sep='\t')

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
            sents_data = fetch_nltk_data(sent_ids)
            pickle.dump(sents_data, f)

    # Instances
    path = os.path.join(DATA_DIR, 'instances.pkl')
    proto_instances = None
    possible = None
    if os.path.exists(path) and not args.init_instances:
        with open(path, 'rb') as f:
            proto_instances, possible = pickle.load(f)
    else:
        with open(path, 'wb') as f:
            proto_instances, possible = build_instance_list(df)
            pickle.dump((proto_instances, possible), f)

    print(f'There are {sum([len(x) for x in proto_instances.values()])} instances.')


    #properties = set(df['Property'].tolist())
    #print(f'There are {len(properties)} many properties')
    #for k, v in proto_instances.items():
    #    for k2, v2 in v.items():
    #        for p in properties:
    #            try:
    #                assert p in v2
    #            except Exception:
    #                print(f'Uh oh! Missing {p}')
    #                print(k2)
    #                print(v2)

    #df2 = df[['Sentence.ID', 'Pred.Token', 'Arg.Pos']]
    #df4 = df[~df2.duplicated()] # (9783, 10), property duplicates removed
    #df5 = df4[['Sentence.ID', 'Pred.Token', 'Arg']]
    #df6 = df4[df5.duplicated()]

    return df, proto_instances, possible, sents_data


if __name__ == '__main__':
    args = get_args()

    df, proto_instances, possible, sents = get_data(args)

    pos = df['Arg.Pos']

    # Junk
    #bools = []
    #bools = [str(p).isdigit() for p in pos]
    #bools_i = [int(x) for x in bools]
    #print(f'{sum(bools)} -- {len(bools)}')


    properties = set(df['Property'].tolist())
    standard_order = ['instigation', 'volition', 'awareness', 'sentient',
    'exists_as_physical', 'existed_before', 'existed_during', 'existed_after',
    'created', 'destroyed', 'predicate_changed_argument', 'change_of_state', 
    'changes_possession', 'change_of_location', 'stationary', 'location_of_event', 
    'makes_physical_contact', 'manipulated_by_another']

    for prop in standard_order:
        train = possible['train'][prop]
        dev = possible['dev'][prop]
        print(f'{prop} -- Train: {train}, Dev: {dev}\n')


    if args.model_type == 'majority':
        model_path = os.path.join(MODEL_DIR, 'majority.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = MajorityBaseline(proto_instances, properties)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)


    trees = sents['trees']
    X_test, y_test = data_utils.get_ins_outs(args, proto_instances['test'],
            properties=properties, trees=trees)

    # Lengths (in tokens) of arguments in test set
    plt.hist([len(x['arg']) for x in X_test])

    breakpoint()

    res = eval.evaluate(args, model, X_test, y_test)

    p = res['tp'] / (res['tp'] + res['fp'])
    r = res['tp'] / (res['tp'] + res['fn'])
    F = (2*r*p) / (r+p)

    print(f'Prec: {p}\tRec: {r}\tF1: {F}')

    print('End of main.')
    breakpoint()

