from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()

    # Could probably collapse these all into one args.init list
    parser.add_argument('-ii', help='Initialize proto-role data?',
            action='store_true', dest='init_instances')
    parser.add_argument('-is', help='Initialize sentence data (NLTK)?',
            action='store_true', dest='init_sents')
    parser.add_argument('-ig', help='Initialize glove data?',
            action='store_true', dest='init_glove')
    parser.add_argument('-id', help='Initialize dependency data?',
            action='store_true', dest='init_deps')

    #parser.add_argument('model', help='Name of model', default='dummyname')
    parser.add_argument('-seed', type=int, dest='seed', default=7)

    parser.add_argument('-lrf', dest='logreg_feats', default=[])

    parser.add_argument('-t', help='Type of model (glove, logreg, etc.)',
            default='majority', dest='model_type')
    parser.add_argument('-ft', help='Features for training', dest='features',
            default=['pred_lemma'], nargs='+')

    parser.add_argument('-lr', help='Learning rate.', dest='lr', type=float, default=1e-3)
    parser.add_argument('-bs', help='Batch size.', type=int, dest='batch_size',
            default=50)
    parser.add_argument('-h1', help='Hidden size.', type=int, dest='h_size',
            default=100)

    parser.add_argument('-epochs', help='Number of epochs in training.',
            type=int, default=1)

    args = parser.parse_args()
    return args
