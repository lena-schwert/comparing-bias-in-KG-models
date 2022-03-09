# %% Imports

import os

from src.utils import set_base_path_based_on_host, improve_pandas_viewing_options
import pandas as pd
import matplotlib.pyplot as plt

BASE_PATH_HOST = set_base_path_based_on_host()
improve_pandas_viewing_options()

# %% What are the most common occupations?

# option A: take the tail value counts based on wikidata5m_all_triplets.txt
tail_value_counts = pd.read_csv('exploration/relationship_counts/tail_value_counts_all_11.11.2021.csv')
# option B: create tail value counts from the human facts subset:wikidata5m_human_facts_subset_complete_050122.tsv
tail_value_counts = pd.read_csv('data/interim/tail_value_counts_human_facts_W5M_8.2.2022.csv')

tail_value_counts = tail_value_counts[tail_value_counts['dataset_name'] == 'Wikidata5M']

# Wikidata target relation: occupation = P106
W5M_occupations = tail_value_counts[tail_value_counts['relation_P_ID'] == 'P106']
W5M_occupations = W5M_occupations.convert_dtypes()
W5M_occupations.reset_index(inplace = True, drop = True)

# make sure the dataset is sorted by counts
W5M_occupations.sort_values(ascending = False,
                            by = 'count', inplace = True)

# Choose the top-k occupations
k = 50
top_k_occupations = W5M_occupations.head(k)

top_k_occupations.to_csv('top_50_occupations_W5M_human_facts_8.2.2022.csv')