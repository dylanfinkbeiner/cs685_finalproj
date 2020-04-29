from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()

    # Could probably collapse these all into one args.init list
    parser.add_argument('-i', help='What shall we initialize?', dest='init_list',
            default=[], nargs='*') # sents, deps, instances, glove, dicts

    parser.add_argument('-apa', help='Add predicate and arg tokens, and arg\
            indices to proto_instances?',
            action='store_true', dest='add_pred_args')    

    parser.add_argument('-pp', help='Print counts of positive instances for\
            each property?',
            action='store_true', dest='print_possible')    

    #parser.add_argument('model', help='Name of model', default='dummyname')
    parser.add_argument('-seed', type=int, dest='seed', default=7)

    parser.add_argument('-lrf', dest='logreg_feats', default=[])

    parser.add_argument('-t', help='Type of model (glove, logreg, etc.)',
            default='majority', dest='model_type')
    parser.add_argument('-ft', help='Features for training', dest='features',
            default=['pred_lemma'], nargs='+')

    parser.add_argument('-lr', help='Learning rate.', dest='lr', type=float, default=1e-3)
    parser.add_argument('-bs', help='Batch size.', type=int, dest='batch_size',
            default=10)
    parser.add_argument('-h1', help='Hidden size.', type=int, dest='h_size',
            default=100)
    parser.add_argument('-gd', help='Dimension of glove embs.', type=str,
            dest='glove_d', default='100')

    parser.add_argument('-e', help='Number of epochs in training.',
            dest='epochs', type=int, default=1)

    args = parser.parse_args()
    return args
