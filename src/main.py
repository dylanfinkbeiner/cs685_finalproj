import pickle
from tqdm import tqdm
from collections import defaultdict
import os
import random

import pandas as pd
from nltk.corpus import ptb
from nltk.tree import ParentedTree
from matplotlib import pyplot as plt

from argparser import get_args
from models import MajorityBaseline
import data_utils
import eval

# Want to figure out a way to define this once across all files
SPLITS = ['train', 'dev', 'test'] 

DATA_DIR = '../pickled_data/'
MODEL_DIR = '../saved_models/'
proto_tsv = '../protoroles_eng_pb_08302015.tsv'



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
            breakpoint()
            pickle.dump((proto_instances, possible), f)

    num_instances = sum([len(x) for x in proto_instances.values()])
    print(f'There are {num_instances} instances.')

    return df, proto_instances, possible, sents_data


if __name__ == '__main__':
    args = get_args()



    df, proto_instances, possible, sents = get_data(args)

    # Get actual pred and arg tokens, then normalize data 
    data_utils.normalize(proto_instances)

    print('Got data.')

    X,y = data_utils.get_ins_outs(proto_instances['train'], args,
            sents['trees'])

    breakpoint()


    # Properties are listed in a particular order in SPRL paper
    properties = set(df['Property'].tolist())
    standard_order = ['instigation', 'volition', 'awareness', 'sentient',
    'exists_as_physical', 'existed_before', 'existed_during', 'existed_after',
    'created', 'destroyed', 'predicate_changed_argument', 'change_of_state', 
    'changes_possession', 'change_of_location', 'stationary', 'location_of_event', 
    'makes_physical_contact', 'manipulated_by_another']

    # TODO Why are these numbers not matching the ones in SPRL paper?
    for prop in standard_order:
        train = possible['train'][prop]
        dev = possible['dev'][prop]
        print(f'{prop} -- Train: {train}, Dev: {dev}\n')


    # TODO Messing around, this should be moved to train.py
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
    X = {}
    y = {}
    for split in SPLITS:
        X_split, y_split = data_utils.get_ins_outs(args, proto_instances['split'],
                properties=properties, trees=trees)
        X['split'] = X_split
        y['split'] = y_split


    # Lengths (in tokens) of arguments in test set
    all_one_X = []
    for s in X.values():
        all_one_X.append(s)
    plt.hist([len(x['arg']) for x in all_one_X])

    results = eval.evaluate(args, model, X_test, y_test)

    # Setting up models and training, evaluating
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('End of main.')
    breakpoint()


