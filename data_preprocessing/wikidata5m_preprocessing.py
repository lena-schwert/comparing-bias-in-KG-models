# %% Imports

import os

from utils import set_base_path_based_on_host, get_triples_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 100)

BASE_PATH_HOST = set_base_path_based_on_host()

# %% Load the raw triples

wikidata5m_raw_triples, _ = get_triples_df('wikidata5m')

# get information about size requirements
wikidata5m_raw_triples.info()  # angeblich "nur" 500 MB

# %% get list of all human entities

human_entities = wikidata5m_raw_triples['head_entity'][
    (wikidata5m_raw_triples['relation'] == 'P31') & (wikidata5m_raw_triples['tail_entity'] == 'Q5')]
human_entities.rename('human_head_entities', inplace = True)

# doublecheck that list of human entities only contains unique entries
human_entities_unique = human_entities.unique()
assert len(human_entities) == len(human_entities_unique), "There are duplicate human entities!"
human_entities = human_entities.reset_index(drop = True)

# SAVE TO TSV FOR MORE STABILITY THAN PICKLE!
# omit the column name and the index!
human_entities.to_csv(
    os.path.join(BASE_PATH_HOST, 'data_preprocessing/wikidata5m_human_entities_040122.tsv'),
    sep = '\t', header = False, index = False)

# %% filter triples_df down to only the triples that have a human head entity

# How do I do that efficiently when my look-up list has 1.5 million rows?
# Can I filter the dataframe directly by using the QIDs?

# try dataframe.isin() - success!
human_triples_subset_mask = wikidata5m_raw_triples['head_entity'].isin(human_entities.values)
human_triples_subset = wikidata5m_raw_triples[human_triples_subset_mask]
# reset index
human_triples_subset = human_triples_subset.reset_index(drop = True)

# SAVE AS TSV
# omit the column name and the index!
human_triples_subset.to_csv(os.path.join(BASE_PATH_HOST,
                                         'data_preprocessing/wikidata5m_human_facts_subset_complete_050122.tsv'),
                            sep = '\t', header = False, index = False)

# %% TODO compare my human entities with those from Keidar repo

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

# %% Explore the human subgraph

relation_counts = human_triples_subset['relation'].value_counts()

# not much too see, too many small counts
relation_counts.plot.bar()
relation_counts[relation_counts >= 10000].plot.bar(
    title = 'Relations of human entities that have \n at least 10,000 occurrences in Wikidata5M')

plt.show()

# %% TODO retrieve gender relations for all human entities

# %% TODO create train, validation, test split

# %% TODO alternative: keep existing splits and use only human facts

# check whether doing this removes triple uniformly from all 3 splits
