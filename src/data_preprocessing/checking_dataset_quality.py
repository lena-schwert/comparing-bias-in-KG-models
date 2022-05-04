# %% Imports

import os

from src.utils import set_base_path_based_on_host, get_triples_df
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)

BASE_PATH_HOST = set_base_path_based_on_host()






# %% TODO Sanity check: compare my human entities with those from Keidar repo

Keidar_entities, Keidar_relations = pd.read_pickle(os.path.join(BASE_PATH_HOST,
                                                                'code_from_other_papers/Keidar_automatic_bias_detec/data/wiki5m/human_ent_rel_sorted_list.pkl'))
Keidar_entities = pd.Series(Keidar_entities)

my_human_ent = pd.read_pickle(
    os.path.join(BASE_PATH_HOST, 'data_preprocessing/wikidata5m_human_entities.pkl'))

### compare the two pandas series

# intersection
ents_in_both_dfs = pd.Series(list(set(my_human_ent) & set(Keidar_entities)))
# only 393,719 are matching!

my_human_ent == Keidar_entities.values