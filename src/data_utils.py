from collections import defaultdict
from collections import Counter
from tqdm import tqdm
from random import shuffle
import os
import pickle

from nltk.corpus import ptb
from nltk.tree import ParentedTree
import StanfordDependencies
from typing import Tuple
import numpy as np

from names import *


class Conllu_Token():
    def __init__(self, line):
        self.word = line[1]
        self.pos = line[4]
        self.head = int(line[6])
        self.rel = line[7]

    def __repr__(self):
        return f'Token({self.word}, {self.pos}, {self.head}, {self.rel})'


def conllu_to_sents(conllu_file: str):
    sents_everything = []

    with open(conllu_file, 'r') as f:
        lines = f.readlines()
    if lines[-1] != '\n':
        lines.append('\n') # So split_points works properly
    while lines[0] == '\n':
        lines.pop(0)

    split_points = [idx for idx, line in enumerate(lines) if line == '\n']

    sent_start = 0
    for sent_end in split_points: # Assumes the final line is '\n'
        sents_everything.append(lines[sent_start: sent_end])
        sent_start = sent_end + 1 # Skipping the line break

    # i indexes the sentence number within the conllu file
    sents_just_tokens = []
    for i, s in enumerate(sents_everything):
        s_split = [line.rstrip().split('\t') for line in s] # list of lists
        s_split = [Conllu_Token(line) for line in s_split] #

        sents_everything[i] = s_split
        sents_just_tokens.append([(x.word).lower() for x in s_split])

    return sents_everything, sents_just_tokens


def match_conllu_to_raw(raw, dependencies):
    sent_ids = raw.keys()

    for sent_id, raw_ptb in raw.items():
        k = 0
        for i, token in enumerate(raw_ptb):
            dep = dependencies[sent_id][i]
            if token != dep.word and token != dep.pos and (not '/' in token):
                dependencies[sent_id].insert(i, token)
                for j, t in enumerate(dependencies[sent_id]):
                    if type(t) == Conllu_Token and t.head >= i:
                        dependencies[sent_id][j].head += 1

# point of this function is to get pred and arg indices into raw tokens to
# match locations in Connlu
def match_raw_to_conllu(proto_instances, raw, deps_just_tokens):
    all_instances = []
    for split in SPLITS:
        all_instances.extend(proto_instances[split])

    # Cheesy
    #sid_to_instances = {sent_id: [] for sent_id in raw.keys()}
    #for pt in all_instances:
    #    sid_to_instances[pt['Sentence.ID']].append(pt)

    def lowered(token_seq):
        return [t.lower() for t in token_seq]

    for pt in tqdm(all_instances, desc='Matching raw to Conllu'):
        sent_id = pt['Sentence.ID']
        deps_sent = deps_just_tokens[sent_id]
        raw_ptb = list(raw[sent_id])
        pred_idx = pt['Pred.Token']
        arg_idxs = pt['arg_indices']
        new_pred_idx = pred_idx # Just making copies
        new_arg_idxs = list(arg_idxs)
        idx_to_remove = []
        j = 0
        for i, raw_token in enumerate(raw_ptb):
            dep_token = deps_sent[j]

            okay = True
            if raw_token.lower() != dep_token:
                if raw_token == '-LRB-':
                    raw_ptb[i] = '('
                elif raw_token == '-RRB-':
                    raw_ptb[i] = ')'
                elif '\\' in raw_token:
                    raw_ptb[i] = raw_ptb[i].replace('\\', '')

                else:
                    okay = False
                    idx_to_remove.append(i)

                    if i < pred_idx:
                        new_pred_idx -= 1
                    if i < arg_idxs[0]:
                        for k in range(len(new_arg_idxs)):
                            new_arg_idxs[k] -= 1
            if okay: 
                j += 1

        cleaned_raw = [t for i,t in enumerate(raw_ptb) if i not in
                idx_to_remove]
        try:
            assert lowered(cleaned_raw) == deps_sent
        except Exception:
            print(f'cleaned_raw: {cleaned_raw}')
            print(f'deps_sent: {deps_sent}')
            breakpoint()
        # Correctly modifying all_instances in-place
        pt['Pred.Token'] = new_pred_idx 
        pt['arg_indices'] = new_arg_idxs

    #updated_instances = {split: list() for split in SPLITS}
    #for pt in all_instances:
    #    updated_instances[pt['Split']].append(pt)
    #breakpoint()

    # We only modified the proto_instances, but we should check to see that
    # this in-place modification holds after the function is finished
    #return updated_instances
    return



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


def add_indices_to_terminals(ptree):
    indexed = ParentedTree.convert(ptree)
    for idx, _ in enumerate(ptree.leaves()):
        tree_location = ptree.leaf_treeposition(idx)
        non_terminal = indexed[tree_location[:-1]]
        if "_" in non_terminal[0]:
            print('NO! There are underscores in PTB!!!')
            breakpoint()
            raise Exception
        else:
            non_terminal[0] = non_terminal[0] + "_" + str(idx)
    return indexed


def get_argument(terminal_id, height, ptree):
    indexed = add_indices_to_terminals(ptree)
    terminals = get_terminals(indexed)

    parent = terminals[terminal_id]
    if height > 0:
        for _ in range(height):
            parent = parent.parent()

    arg_leaves = parent.leaves()
    #sent_leaves = ptree.leaves()

    #breakpoint()

    token_idx_pairs = [leaf.split('_') for leaf in arg_leaves]
    arg = [t_i[0] for t_i in token_idx_pairs]
    arg_idx = [int(t_i[1]) for t_i in token_idx_pairs]

    #breakpoint()

    # Seems like a clunky way to get indices
    # XXX DOES NOT WORK, LEAF == . MIGHT BE TRUE COINCIDENTALLY (as in 'THE')
    #arg_idx = []
    #j = 0
    #for i, leaf in enumerate(sent_leaves):
    #    if leaf == arg_leaves[j]:
    #        arg_idx.append(i)
    #        j += 1
    #    if j >= len(arg_leaves):
    #        break

    return arg, arg_idx


# NOTE Kind of a heuristic, seems to work fine though
def get_terminals(ptree: ParentedTree) -> list:
    terms = ptree.subtrees(filter=lambda x: len(list(x.subtrees())) == 1)
    terms = list(terms)
    assert len(ptree.leaves()) == len(terms) # Pull out to unit test?

    return terms


# Properties is just a list of proto-properties, for ease of work
def get_ins_outs(args, data_points, properties=None, sents=None, w2e=None) -> Tuple[list,
        list]:
    possible_feats = ['pred_lemma_emb', 'pred_pos', 'arg_direction']

    acceptable_rels = ['nsubj', 'dobj', 'nmod', 'nsubjpass', 'nmod:npmod',
            'compound:prt', 'iobj', 'advcl', 'xcomp', 'advmod', 'conj',
            'nmod:tmod', 'ccomp', 'dep']
    #acceptable_rels = ['nsubj', 'dobj']

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
            no_glove = []
            for pt in tqdm(data_points, desc="Getting features"):
                X_d = {}
                sent_id = pt['Sentence.ID']

                if 'pred_lemma' in args.features:
                    X_d['pred_lemma'] = pt['pred_lemma']
                if 'pred_lemma_emb' in args.features:
                    if pt['pred_lemma'] in w2e:
                        X_d['pred_emb'] = w2e[pt['pred_lemma']]    
                    else:
                        print(f"No embedding for {pt['pred_lemma']}")
                        no_glove.append(pt['pred_lemma'])
                        X_d['pred_emb'] = np.zeros(100)
                if 'arg_direction' in args.features:
                    if pt['Pred.Token'] < pt['arg_indices'][0]:
                        X_d['arg_direction'] = 0
                    elif pt['Pred.Token'] > pt['arg_indices'][0]:
                        X_d['arg_direction'] = 1
                    else:
                        raise Exception
                if 'arg_distance' in args.features:
                    arg_idx = pt['arg_indices']
                    arg_mid = (arg_idx[0] + arg_idx[-1]) / 2
                    X_d['arg_distance'] = abs(arg_mid - float(pt['Pred.Token']))
                if 'arg_rel' in args.features:
                    dep = sents['dependencies'][sent_id]
                    pred_idx = pt['Pred.Token']
                    arg_idx = pt['arg_indices']
                    arg_rel = None
                    for idx in arg_idx:
                        if type(dep[idx]) == Conllu_Token:
                            if dep[idx].head == pred_idx + 1:
                                #X_d['arg_rel'] = dep[idx].rel
                                if dep[idx].rel in acceptable_rels:
                                    arg_rel = dep[idx].rel
                                    break
                            elif dep[idx].head == dep[pred_idx].head and dep[pred_idx].rel == 'conj':
                                if dep[idx].rel in acceptable_rels:
                                    arg_rel = dep[idx].rel
                                    break
                    if arg_rel == None:
                        X_d['arg_rel'] = '<other_or_no>'
                    else:
                        X_d['arg_rel'] = arg_rel

                X.append(X_d) # (property, feats)
                y.append({p:pt[p]['binary'] for p in properties})

    return X, y


def get_ins_outs_lstm(instances, numericalized):
    X = []
    y = []

    for pt in instances:
        pred_idx, head_idx = pt['Pred.Token'], pt['arg_head_idx'] # ints
        curr_x = (numericalized[pt['Sentence.ID']], (pred_idx, head_idx)) # np array of ints
        X.append(curr_x)
        y.append(np.array([pt[p]['binary'] for p in PROPERTIES]))

    return X, y


# Specifically builds the numpy array of word embeddings, such that index
# along 0th dim corresponds to indices in w2i
def build_emb_np(w2e, w2i=None, i2w=None):
    dict_length = len(w2e)

    emb_size = len(w2e['the']) # Should be the same length as any other word vector
    def rand_vect():
        vec = np.random.randn(emb_size)
        return (vec / np.linalg.norm(vec)).tolist()

    emb_list = [rand_vect()] * 2

    # Important that we append embs in order of indices i
    words = [w for w,i in sorted(w2i.items(), key=lambda x: x[1])]
    words.remove(PAD_TOKEN) 
    words.remove(UNK_TOKEN) 
    for w in tqdm(words, ascii=True, desc=f'Building emb_np', ncols=80):
        emb_list.append(w2e[w])

    emb_np = np.array(emb_list)

    return emb_np


def get_vocabulary(deps_just_tokens, sent_ids=None):
    # deps_just_tokens[sent_id] = list of lowercased tokens
    vocab = set()
    #NOTE We're building vocab with all splits, and the GloVe embbeddings will
    #dictate which tokens ought to be unks; otherwise we'd use sent_ids to
    #select build vocab specifically sents with id in sent_ids['train']
    for sent in deps_just_tokens.values():
        for token in sent:
            vocab.add(token)
    return vocab


#TODO Unused right now?
def clean_traces(raw_sent):
    cleaned = []
    for token in raw_sent:
        if '*' in token:
            continue
        else:
            cleaned.append(token)
    return cleaned


def unkify_logreg_features(X, cutoff=0):
    features = set(X['train'][0].keys()) # X[0] should be as good as any other

    X_train = X['train']
    counts = get_feature_counts(X_train)
    #for k, v in sorted(counts['arg_rel'].items(), key=lambda item: item[1]):
    #    print(f"{k}: {v}\n")
    #breakpoint()


    #exceptions = {'arg_direction', 'arg_distance'}
    # TODO Should define exceptions in terms of the data type of feat
    exceptions = {'arg_direction', 'arg_distance', 'pred_lemma_emb'}
    for split in SPLITS:
        for pt in X[split]:
            for f in features - exceptions: 
                if f in counts and counts[f][pt[f]] <= cutoff:
                    pt[f] = f'<unk_{f}>'

    return


def get_feature_counts(X):
    features = X[0].keys() # X[0] should be as good as any other

    acceptable_types = [str, int]
    counts = {} # {feature : Counter}
    for f in features:
        # Round up the set of all values feat takes on in train set
        if type(X[0][f]) in acceptable_types:
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

    # First, get the predicate and arg tokens
    for split in SPLITS:
        data_points = proto_instances[split]
        for pt in tqdm(data_points, desc=f"Get pred, args, for {split}"):
            s_id = pt['Sentence.ID']
            tree = trees[s_id]
            ptree = ParentedTree.convert(tree)
            pred, arg, arg_idx = get_pred_arg(pt, ptree)

            pt['pred_token'] = pred
            pt['arg_tokens'] = arg
            pt['arg_indices'] = arg_idx

    return


def normalize(deps_just_tokens, args):
    # TODO
    return

def unkify():
    # TODO
    return

def numericalize(deps_just_tokens, w2i):
    numericalized = {}
    for sent_id, sent in deps_just_tokens.items():
        ints = [w2i.get(tok, w2i[UNK_TOKEN]) for tok in sent]
        ints = np.array(ints)
        numericalized[sent_id] = ints

    return numericalized


def get_arg_head_idx(proto_instances, dependencies, deps_just_tokens):
    for curr_split in proto_instances.values():
        for pt in curr_split:
            sent_id = pt['Sentence.ID']
            dep = dependencies[sent_id]
            # From left to right, find first token whose
            # head is not in the dominated constituent (i.e. internal
            # to the argument span)
            head_idx = None
            arg_idxs = pt['arg_indices']
            for idx in arg_idxs:
                conll_tok = dep[idx]
                if conll_tok.head - 1 in arg_idxs:
                    continue
                else:
                    head_idx = idx
                    break
            try:
                assert head_idx != None
            except Exception:
                print(f'Head_idx was not found in sentence:')
                print(deps_just_tokens[sent_id])
            pt['arg_head_idx'] = head_idx

    return # modified in place?


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


def get_dependencies(sent_ids):
    deps = {}
    deps_just_tokens = {}

    def parse_id(sent_id):
        return sent_id.split('_')

    # NOTE Didn't work because of some issue with JPype1
    #sd = StanfordDependencies.get_instance(backend='subprocess')
    #sd = StanfordDependencies.get_instance()
    #for sid, tree in tqdm(trees.items(), desc='Converting to dependencies'):
    #    dependencies[sid] = sd.convert_tree(str(tree))

    mrg_ids = [parse_id(sid)[0] for sid in sent_ids]

    conllus = {}
    conllus_just_tokens = {}
    for mrg_id in tqdm(mrg_ids, desc="Getting conllus"):
        path = os.path.join(CONLLU_DIR, mrg_id[:2], f'WSJ_{mrg_id}.conllu')
        sents_everything, sents_just_tokens = conllu_to_sents(path)
        conllus[mrg_id] = sents_everything 
        conllus_just_tokens[mrg_id] = sents_just_tokens

    for sent_id in sent_ids:
        mrg_id, sent_num = parse_id(sent_id)
        deps[sent_id] = conllus[mrg_id][int(sent_num)]
        deps_just_tokens[sent_id] = conllus_just_tokens[mrg_id][int(sent_num)]

    return deps, deps_just_tokens


def build_dicts(deps_just_tokens, sent_ids=None, glove_vocab=None):
    word = set() 
    #for sent_id, sent in deps_just_tokens.items():
    #    if sent_id in sent_ids['train']:
    #        for token in sent:
    #            if word in glove_vocab:
    #            word.add(sent)
    word = sorted(glove_vocab)

    w2i = defaultdict(lambda : len(w2i))
    i2w = dict()

    i2w[w2i[PAD_TOKEN]] = PAD_TOKEN
    i2w[w2i[UNK_TOKEN]] = UNK_TOKEN
    #i2w[w2i[ROOT_TOKEN]] = ROOT_TOKEN

    for w in word:
        i2w[w2i[w]] = w

    return dict(w2i), i2w


def build_instance_list(df):
    print(f'df has {df.shape[0]} entries')

    instances = {name: dict() for name in SPLITS}

    prop_cols = ['Response', 'Applicable']
    cols = set(df.columns.tolist())
    not_props = cols - set(prop_cols) - {'Property'}

    properties = list(set(df['Property'].tolist()))
    possible = {split:{p:0 for p in properties} for split in ['train', 'dev']}

    for _, x in tqdm(df.iterrows(), desc='Building instances list'):
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

        d[x['Property']] = {c: x[c] for c in prop_cols}
        binary_label = int(x['Applicable'] and (x['Response'] >= 4))
        #binary_label = x['Response'] >= 4 # Doesn't make a difference
        d[x['Property']]['binary'] = binary_label
        
        if binary_label == 1 and x['Split'] != 'test':
            possible[x['Split']][x['Property']] += 1

    # ID keys serve no further purpose, listify the splits
    for split in SPLITS:
        data_points = instances[split]
        listified = sorted(data_points.items(), key=lambda x: x[0])
        instances[split] = [x[1] for x in listified]

    return instances, possible


def w2e_from_file(emb_file, vocab=None):
    w2e = {}
    glove_vocab = []

    print(f'Building w2e from file: {emb_file}')

    # XXX Random would be better?
    unk = [0.] * 100

    with open(emb_file, 'r', errors='ignore') as f:
        lines = f.readlines()

        if len(lines[0].split()) == 2:
            lines.pop(0)

        for i, line in tqdm(enumerate(lines), ascii=True, desc=f'w2e Progress', ncols=80):
            #if i >= 1000000:
            #    break
            line = line.split()
            word = line[0].lower()
            if word not in vocab:
                continue
            try:
                word_vector = [float(value) for value in line[1:]]
                w2e[word] = np.array(word_vector)
                glove_vocab.append(word)
            except Exception:
                print(f'Word is: {word}, line is {i}')

    return w2e 

