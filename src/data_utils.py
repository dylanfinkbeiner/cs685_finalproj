from collections import defaultdict
from collections import Counter
from tqdm import tqdm
from random import shuffle

from nltk.corpus import ptb
from nltk.tree import ParentedTree
from typing import Tuple


import numpy as np


from names import *
#SPLITS = ['train', 'dev', 'test'] 
#GLOVE_PKL = '../pickled_data/glove.pkl'


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
        if len(args.features) == 0:
            print('Need to specify at least one feature.')
            raise Exception
        else:
            if 'pred_emb' in args.features:
                path = os.path.join(PICKLED_DIR, 'glove.pkl')
                if os.path.exists(path) and not args.init_glove:
                    with open(path, 'rb') as f:
                        w2e = pickle.load(f)
                else:
                    w2e = w2e_from_file(os.path.join(GLOVE_FILE)
                    with open(path, 'wb') as f:
                        pickle.dump(w2e, f)

            for pt in data_points:
                # One training example per property
                X_d = {}

                #for feat in args.features:
                    #add_feature(feat, X_d, pt, sents)
                
                if 'pred_lemma' in args.features:
                    X_d['pred_lemma'] = pt['pred_lemma']
                if 'pred_lemma_emb' in args.features:
                    if pt['pred_lemma'] in w2e:
                        X_d['pred_emb'] = w2e[pt['pred_lemma']]    
                    else:
                        print(f"No embedding for {pt['pred_lemma']}")
                        raise Exception
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

                X.append(X_d) # (property, feats)
                y.append({p:pt[p]['binary'] for p in properties})

    return X, y


def unkify(X, cutoff=1):
    features = set(X['train'][0].keys()) # X[0] should be as good as any other

    X_train = X['train']
    counts = get_counts(X_train)

    #exceptions = {'arg_direction', 'arg_distance'}
    # TODO Should define exceptions in terms of the data type of feat
    exceptions = {'arg_direction', 'arg_distance'}
    for split in SPLITS:
        for pt in X[split]:
            for f in features - exceptions: 
                if counts[f][pt[f]] <= cutoff:
                    pt[f] = f'<unk_{f}>'

    return


def get_counts(X):
    features = X[0].keys() # X[0] should be as good as any other

    counts = {} # {feature : Counter}
    for f in features:
        # Round up the set of all values feat takes on in train set
        values = [pt[f] for pt in X]
        counts[f] = Counter(values)

    return counts


def split_on_properties(y, properties=[]):
    y_p = {p: list() for p in properties}
    for y_curr in y:
        for p in properties:
            y_p[p].append(y_curr[p])

    return y_p # y_p[p] = 9378-long list of labels


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


def w2e_from_file(emb_file):
    w2e = {}

    print(f'Building w2e from file: {emb_file}')

    with open(emb_file, 'r', errors='ignore') as f:
        lines = f.readlines()

        if len(lines[0].split()) == 2:
            lines.pop(0)

        for i, line in tqdm(enumerate(lines), ascii=True, desc=f'w2e Progress', ncols=80):
            #if i >= 200000:
            if i >= 10000:
                break
            line = line.split()
            word = line[0].lower()
            try:
                word_vector = [float(value) for value in line[1:]]
                w2e[word] = word_vector
            except Exception:
                print(f'Word is: {word}, line is {i}')

    return w2e

