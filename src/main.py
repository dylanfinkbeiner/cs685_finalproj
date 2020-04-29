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


def recreate_created_destroyed(proto_instances):
    created = {'train': 0, 'dev': 0}
    destroyed = {'train': 0, 'dev': 0}

    for split in proto_instances.keys():
        if split == 'test':
            continue
        for pt in proto_instances[split]:
            if pt['existed_before']['binary'] == 1:
                if pt['existed_after']['binary'] == 0:
                    destroyed[split] += 1
                    try:
                        assert pt['destroyed']['binary'] == 1
                    except Exception:
                        print('Not destroyed!')
                        breakpoint()
            else:
                if pt['existed_after']['binary'] == 1:
                    created[split] += 1
                    #assert pt['created'] == 1

    return created, destroyed


def stationary_v_location(proto_instances, raw):
    compare = {'inverse': 0, 'not': 0}

    for split in proto_instances.keys():
        for pt in proto_instances[split]:
            location = pt['change_of_location']['binary']
            stationary = pt['stationary']['binary']
            if location != stationary:
                compare['inverse'] += 1
                    #try:
                    #    assert pt['destroyed']['binary'] == 1
                    #except Exception:
                    #    print('Not destroyed!')
                    #    breakpoint()
            elif location == 1 and stationary == 1:
                print('Huh??')
                pred = pt['pred_token']
                arg = pt['arg_tokens']
                sent = raw[pt['Sentence.ID']]
                print(f'Predicate {pred}, arg {arg}')
                breakpoint()
                compare['not'] += 1
                    #assert pt['compare'] == 1

    return compare


# What other properties occur with change_of_location?
def locations(proto_instances, raw):
    others = {p:0 for p in PROPERTIES}

    for split in proto_instances.keys():
        for pt in proto_instances[split]:
            location = pt['location_of_event']['binary']
            if location:
                # Fact that locations are so often embedded inside
                # arguments may explain low interann agreement
                if not pt['exists_as_physical']['binary']:
                    print('A non-physical location??')
                    pred = pt['pred_token']
                    arg = pt['arg_tokens']
                    sent = raw[pt['Sentence.ID']]
                    print(f'Predicate {pred}, arg {arg}')
                    print(sent)
                    print(pt['location_of_event'])
                    print(pt['exists_as_physical'])
                    breakpoint()
                for p in PROPERTIES:
                    if pt[p]['binary']:
                        others[p] += 1

    values = sorted(others.items(), key = lambda x: x[0])
    values = [x[1] for x in values]
    props = sorted(list(others.keys()))
    plt.bar(x=np.arange(len(values)), height=values, tick_label=props)
    plt.xticks(rotation=90)
    plt.show()
    return 


def get_data(args):
    df = pd.read_csv(PROTO_TSV, sep='\t')

    # Sentences
    sent_ids = set(df['Sentence.ID'].tolist())
    print(f'There are {len(sent_ids)} unique sentences.')
    sents_path = os.path.join(PICKLED_DIR, 'sents.pkl')
    sents = None
    if os.path.exists(sents_path) and not 'sents' in args.init_list:
        with open(sents_path, 'rb') as f:
            sents = pickle.load(f)
    else:
        with open(sents_path, 'wb') as f:
            sents = data_utils.get_nltk_sents(sent_ids)
            pickle.dump(sents, f)

    # Dependency data
    dependencies_path = os.path.join(PICKLED_DIR, 'dependencies.pkl')
    if os.path.exists(dependencies_path) and not 'deps' in args.init_list:
        with open(dependencies_path, 'rb') as f:
            deps, deps_just_tokens = pickle.load(f)
    else:
        with open(dependencies_path, 'wb') as f:
            deps, deps_just_tokens = data_utils.get_dependencies(sent_ids)
            pickle.dump((deps, deps_just_tokens), f)
    sents['dependencies'] = deps
    sents['deps_just_tokens'] = deps_just_tokens

    # Instances
    instances_path = os.path.join(PICKLED_DIR, 'instances.pkl')
    proto_instances = None
    possible = None # Data to compare to SPRL paper
    if os.path.exists(instances_path) and not 'instances' in args.init_list:
        with open(instances_path, 'rb') as f:
            proto_instances, possible = pickle.load(f)
    else:
        proto_instances, possible = data_utils.build_instance_list(df)
        data_utils.add_pred_args(proto_instances, sents['trees'])
        with open(instances_path, 'wb') as f:
            pickle.dump((proto_instances, possible), f)

    # Matching between raw and dependency data
    if args.model_type != 'lstm':
        data_utils.match_conllu_to_raw(sents['raw'], deps)
        # no corresponding overwrite here since running logreg on local machine
    else:
        data_utils.match_raw_to_conllu(
                proto_instances, sents['raw'], deps_just_tokens)
        with open(instances_path, 'wb') as f:
            pickle.dump((proto_instances, possible), f)

    # Word embedding data
    sent_ids = {} # Redefining sent_ids for this section
    for split in SPLITS:
        sent_ids[split] = [pt['Sentence.ID'] for pt in proto_instances[split]]
    w2e = None
    glove_path = os.path.join(PICKLED_DIR, f'glove_{args.glove_d}.pkl')
    if os.path.exists(glove_path) and not 'glove' in args.init_list:
        with open(glove_path, 'rb') as f:
            w2e = pickle.load(f)
    else:
        vocab = data_utils.get_vocabulary(deps_just_tokens)
        w2e = data_utils.w2e_from_file(
                GLOVE_FILE[args.glove_d], vocab=vocab)
        with open(glove_path, 'wb') as f:
            pickle.dump(w2e, f)

    w2i, i2w = None, None
    emb_np = None
    X, y = None, None
    if args.model_type == 'lstm':
        dicts_path = os.path.join(PICKLED_DIR, 'dicts.pkl')
        if os.path.exists(dicts_path) and not 'dicts' in args.init_list:
            with open(dicts_path, 'rb') as f:
                w2i, i2w = pickle.load(f)
        else:
            w2i, i2w = data_utils.build_dicts(
                    sents['deps_just_tokens'],
                    sent_ids=sent_ids,
                    glove_vocab=sorted(list(w2e.keys())))
            with open(dicts_path, 'wb') as f:
                pickle.dump((w2i, i2w), f)
        
        emb_np_path = os.path.join(PICKLED_DIR, 'emb_np.pkl')
        if os.path.exists(emb_np_path) and not 'emb_np' in args.init_list:
            with open(emb_np_path, 'rb') as f:
                emb_np = pickle.load(f)
        else:
            emb_np = data_utils.build_emb_np(w2e, w2i=w2i, i2w=i2w)
            with open(emb_np_path, 'wb') as f:
                pickle.dump(emb_np, f)
        
        lstm_data_path = os.path.join(PICKLED_DIR, 'lstm_data.pkl')
        if os.path.exists(lstm_data_path) and not 'lstm_data' in args.init_list:
            with open(lstm_data_path, 'rb') as f:
                X, y = pickle.load(f)
        else:
            # Proto instances modified in-place here
            data_utils.get_arg_head_idx(proto_instances, sents['dependencies'],
                    sents['deps_just_tokens'])
            with open(instances_path, 'wb') as f:
                pickle.dump((proto_instances, possible), f)

            numericalized = data_utils.numericalize(sents['deps_just_tokens'], w2i)
            X = {}
            y = {}
            for split in SPLITS:
                X[split], y[split] = data_utils.get_ins_outs_lstm(
                        proto_instances[split], numericalized)
            with open(lstm_data_path, 'wb') as f:
                pickle.dump((X, y), f)


    num_instances = sum([len(x) for x in proto_instances.values()])
    print(f'There are {num_instances} instances.')

    return {'df': df, 
            'proto_instances': proto_instances, 
            'possible': possible,
            'sents': sents,
            'w2e': w2e,
            'sent_ids': sent_ids,
            'lstm_data': (X,y),
            'dicts': (w2i, i2w),
            'emb_np': emb_np}



if __name__ == '__main__':
    args = get_args()

    # Things that do not change depending on experiment
    data = get_data(args)
    df = data['df']
    proto_instances = data['proto_instances']
    possible = data['possible']
    sents = data['sents']
    w2e = data['w2e']
    sent_ids = data['sent_ids']
    breakpoint()
    exit()

    # Analyze POS tags of predicate verbs of proto role dataset
    #tagged = sents['tagged']
    #all_pos = []
    #for pt in proto_instances['train']:
    #    # This does not account for multiple pred-arg pairs with same sentence
    #    # and predicate
    #    sid = pt['Sentence.ID']
    #    predid = pt['Pred.Token']
    #    tagged_sent = tagged[sid]
    #    all_pos.append(tagged_sent[predid][1])

    #breakpoint()

    raw = sents['raw']
    #created, destroyed = recreate_created_destroyed(proto_instances)
    #compare = stationary_v_location(proto_instances, sents['raw'])
    #locations(proto_instances, raw)


    # Now normalize (which might be different per experiment)
    #data_utils.normalize(proto_instances, args)

    # TODO Why are these numbers not matching the ones in SPRL paper?

    if args.print_possible:
        array = []
        for p in PROPERTIES:
            possible_train = possible['train'][p]
            possible_dev = possible['dev'][p]
            array.append([p, possible_train, possible_dev])
            print(f'{p} -- Train: {possible_train}, Dev: {possible_dev}\n')
        #array = np.array(array)
        #columns=['name', 'train', 'dev']
        #dfp = pd.DataFrame(array, index=list(range(18)), columns=columns)
        #with open('./tables.txt', 'w') as f:
        #    f.write(dfp.to_latex())

    #STATS

    #arg_lens = []
    #for split in SPLITS:
    #    for case in proto_instances[split]:
    #        if len(case['arg_tokens']) == 49:
    #            print('Wham-o!')
    #            breakpoint()
    #        arg_lens.append(len(case['arg_tokens']))
    #    #len_med = np.median(arg_lens)
    #    #len_m = np.mean(arg_lens)

    #len_med = np.median(arg_lens)
    #len_m = np.mean(arg_lens)
    #breakpoint()

    
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
    elif args.model_type == 'lstm':
        w2i, i2w = data['dicts']
        emb_np = data['emb_np']
        X, y = data['lstm_data']

        model = models.LSTM(
                vocab_size=len(w2i),
                emb_size=int(args.glove_d),
                h_size=args.h_size,
                padding_idx=w2i[PAD_TOKEN],
                emb_np=emb_np,
                properties=PROPERTIES)

        print('Finished building lstm model!')


    elif args.model_type == 'logreg':
        X = {}
        y = {}
        for split in SPLITS:
            X_split, y_split = data_utils.get_ins_outs(args, proto_instances[split],
                    properties=PROPERTIES, sents=sents, w2e=w2e)
            X[split] = X_split
            y[split] = y_split

        print('Got logreg features.')

        data_utils.unkify_logreg_features(X, cutoff=0)

        all_X = []
        l_u = {}
        l = 0
        u = 0
        embs = []
        for split in SPLITS:
            all_X.extend(X[split])
            u += len(X[split])
            if 'pred_lemma_emb' in args.features:
                embs.extend([d.pop('pred_emb') for d in all_X[l:u]])
            l_u[split] = (l, u)
            l = u

        # STATS
        arg_ds= set([x['pred_lemma'] for x in all_X])
        breakpoint()

        v = DictVectorizer(sparse=False)
        vectorized = v.fit_transform(all_X)
        if 'pred_lemma_emb' in args.features:
            embs = np.array(embs)
            vectorized = np.concatenate((vectorized, embs), axis=-1)
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

    metrics = train.train(args, model, X, y)

    array = []
    for k, v in metrics.items():
        array.append([f"{v['F']*100:.2f}", f"{v['precision']*100:.2f}", f"{v['recall']*100:.2f}"])
        print(f"{k}: F={v['F']*100:.2f}, p={v['precision']*100:.2f}, r={v['recall']*100:.2f}")
    array = np.array(array)
    columns = ['F1', 'Precision', 'Recall']
    dfm = pd.DataFrame(array, index=PROPERTIES + ['micro average'], columns=columns)
    breakpoint()
    with open('./tables.txt', 'w') as f:
        f.write(dfm.to_latex())

    print('Finished training.')
    breakpoint()

    results = evaluate.evaluate(args, model, X['test'], y['test'])

    print('End of main.')
    breakpoint()


