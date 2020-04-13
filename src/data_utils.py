from collections import defaultdict
from tqdm import tqdm
from random import shuffle
from nltk.corpus import ptb
from nltk.tree import ParentedTree

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import numpy as np


def get_predicate(terminal_id, tree):
    return tree.leaves()[terminal_id]


def get_terminals(ptree:ParentedTree):
    return list(ptree.subtrees(
        filter=lambda x: len(list(x.subtrees())) == 1))


def get_argument(terminal_id, height, ptree):
    # At least in theory, should be a list of all 
    #terminals = list(ptree.subtrees(
    #    filter=lambda x: len(list(x.subtrees())) == 1))
    terminals = get_terminals(ptree)
    ''' An easy unit test would be to compare ptree.leaves() and [x.leaves() for
        x in terminals
    '''

    parent = terminals[terminal_id]
    if height > 0:
        for _ in range(height):
            parent = parent.parent()

    return parent.leaves()


def get_ins_outs(args, data_points, properties=None, trees=None):
    X = []
    y = []

    data_points = list(data_points.values())
    if args.xy == 'minimal':
        for pt in data_points:
            for p in properties:
                X.append(p)
                y.append(pt[p]['binary'])

    # Dict vectorizer?
    if args.xy == 'logreg': #XXX this is temporary, doesnt make sense
        #v = DictVectorizer(sparse=False)
        #counts = countify(normalized)
        #vectorized = v.fit_transform(counts)
        ## Turn into array so we can do fancy indexing to get top 10
        #names = np.array(v.get_feature_names())
        for pt in data_points:
            s_id = pt['Sentence.ID']
            tree = trees[s_id]
            ptree = ParentedTree.convert(tree)

            pred = get_predicate(pt['Pred.Token'], tree)

            arg = []
            arg_height_pairs = [t.split(':') for t in pt['Arg.Pos'].split(',')]
            for pair in arg_height_pairs:
                arg_pos, height = [int(n) for n in arg_height_pairs[0]]
                arg.extend(get_argument(arg_pos, height, ptree))
            #arg = ' '.join(arg)

            for p in properties:
                d = {
                        'pred': pred, 
                        'pred_lemma': pt['Roleset'].split('.')[0],
                        'arg': arg, 
                        'pb_arg': pt['Arg'],
                        'property': p
                        }
                X.append(d)
                y.append(pt[p]['binary'])


    return X, y



def top_ten_logreg(clf, names):
    coef = clf.coef_.flatten()

    # Top ten for impersonal (0-label)
    idx = np.argsort(coef)
    print(f'Top 10 personal: {names[idx][-10:]}')
    print(f'Top 10 impersonal: {names[idx][:10]}')
