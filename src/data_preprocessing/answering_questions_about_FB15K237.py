# %% import from built-in modules
import os
import sys
import time
import gc
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns

from SPARQLWrapper import SPARQLWrapper, JSON

from src.utils import set_base_path_based_on_host

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 100)

BASE_PATH_HOST = set_base_path_based_on_host()

trainset = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                    'data/processed/files_per_model/KG_and_LM_KGBERT_FB15K-237/train.tsv'),
                       sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])
valset = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                  'data/processed/files_per_model/KG_and_LM_KGBERT_FB15K-237/dev.tsv'),
                     sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])
testset = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                   'data/processed/files_per_model/KG_and_LM_KGBERT_FB15K-237/test_complete.tsv'),
                      sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

entire_dataset = pd.concat([trainset, valset, testset])

# entire_dataset.to_csv(os.path.join(BASE_PATH_HOST, 'data/processed/output_of_preprocessing/FB15K237_complete.tsv'),
#                                    sep = '\t', header = False, index = False)


# %% What are the equivalents of Wikidata relations?
# How many relations are there describing people's properties that I could use for bias measurement?

# search for unique set of relations starting with /people/person/

set_of_person_relations_mask = entire_dataset['relation'].str.contains('/people/person/')
set_of_person_relations = set(entire_dataset['relation'][set_of_person_relations_mask].unique())

# this results in 12 relations!
# {'/people/person/languages',
# '/people/person/spouse_s./people/marriage/type_of_union',
# '/people/person/profession',
# '/people/person/gender',
# '/people/person/sibling_s./people/sibling_relationship/sibling',
# '/people/person/places_lived./people/place_lived/location',
# '/people/person/nationality',
# '/people/person/employment_history./business/employment_tenure/company',
# '/people/person/religion',
# '/people/person/place_of_birth',
# '/people/person/spouse_s./people/marriage/location_of_ceremony',
# '/people/person/spouse_s./people/marriage/spouse'}

# %% Subset testset for bias evaluation using KG-BERT

relations_to_extract = ['/people/person/profession', '/people/person/gender']

only_gender_occupation_facts_mask = testset['relation'].isin(relations_to_extract)
only_gender_occupation_facts = testset[only_gender_occupation_facts_mask]

assert len(relations_to_extract) == len(only_gender_occupation_facts['relation'].unique())

only_gender_occupation_facts.to_csv(
    os.path.join(BASE_PATH_HOST, 'data/processed/output_of_preprocessing/FB15K237_testset_only_gender_occupation_facts.tsv'),
sep = '\t', header = False, index = False)

