import os

DATA_DIR = '../data/'
PICKLED_DIR = os.path.join(DATA_DIR, 'pickled/')
CONLLU_DIR = os.path.join(DATA_DIR, 'WSJ_conllus/')
MODEL_DIR = '../saved_models/'

PROTO_TSV = os.path.join(DATA_DIR, 'protoroles_eng_pb_08302015.tsv')
GLOVE_FILE = os.path.join(DATA_DIR, 'glove.6B.100d.txt')

SPLITS = ['train', 'dev', 'test'] 

PROPERTIES = ['instigation', 'volition', 'awareness', 'sentient',
'exists_as_physical', 'existed_before', 'existed_during', 'existed_after',
'created', 'destroyed', 'predicate_changed_argument', 'change_of_state', 
'changes_possession', 'change_of_location', 'stationary', 'location_of_event', 
'makes_physical_contact', 'manipulated_by_another']

