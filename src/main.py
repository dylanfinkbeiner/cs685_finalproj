import pandas as pd
import pickle
from nltk.corpus import ptb
from tqdm import tqdm
from collections import defaultdict
import os

from argparser import get_args

DATA_DIR = '../pickled_data/'
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

    for _, x in tqdm(df.iterrows(), desc='Processing proto-role data entries:'):
        idstr = [x['Sentence.ID'], x['Pred.Token'], x['Arg'], x['Arg.Pos']]
        idstr = '_'.join([str(n) for n in idstr])

        if idstr not in instances[x['Split']]:
            d = {}
            for col in not_props:
                d[col] = x[col]
            instances[x['Split']][idstr] = d
        else:
            d = instances[x['Split']][idstr]

        d[x['Property']] = {p: x[p] for p in props}


    return instances


def get_data(args):
    df = pd.read_csv(proto_tsv, sep='\t')

    # Sentences
    sent_ids = set(df['Sentence.ID'].tolist())
    print(f'There are {len(sent_ids)} unique sentences.')
    path = os.path.join(DATA_DIR, 'sents.pkl')
    sents_data = None
    if os.path.exists(path) and not args.init_data:
        with open(path, 'rb') as f:
            sents_data = pickle.load(f)
    else:
        with open(path, 'wb') as f:
            sents_data = fetch_nltk_data(sent_ids)
            pickle.dump(sents_data, f)

    # Instances
    path = os.path.join(DATA_DIR, 'instances.pkl')
    proto_instances = None
    if os.path.exists(path) and not args.init_data:
        with open(path, 'rb') as f:
            proto_instances = pickle.load(f)
    else:
        with open(path, 'wb') as f:
            proto_instances = build_instance_list(df)
            pickle.dump(proto_instances, f)

    print(f'There are {sum([len(x) for x in proto_instances.values()])} instances.')


    properties = set(df['Property'].tolist())
    print(f'There are {len(properties)} many properties')
    for k, v in proto_instances.items():
        for k2, v2 in v.items():
            for p in properties:
                try:
                    assert p in v2
                except Exception:
                    print(f'Uh oh! Missing {p}')
                    print(k2)
                    print(v2)
                    breakpoint()

    df2 = df[['Sentence.ID', 'Pred.Token', 'Arg.Pos']]
    df4 = df[~df2.duplicated()] # (9783, 10), property duplicates removed
    df5 = df4[['Sentence.ID', 'Pred.Token', 'Arg']]
    df6 = df4[df5.duplicated()]
    breakpoint()

    return df, proto_instances, sents_data


if __name__ == '__main__':
    args = get_args()

    df, instances, sents = get_data(args)



