# %% Imports

import os

from src.utils import set_base_path_based_on_host, improve_pandas_viewing_options
import pandas as pd
import matplotlib.pyplot as plt

BASE_PATH_HOST = set_base_path_based_on_host()
improve_pandas_viewing_options()


# %% What are the most frequently occurring relations in the human subgraph?

pd.read_csv('wikidata5m_human_facts_subset_complete_050122.tsv')

relation_counts = human_triples_subset['relation'].value_counts()

# not much too see, too many small counts
relation_counts.plot.bar()
relation_counts[relation_counts >= 10000].plot.bar(
    title = 'Relations of human entities that have \n at least 10,000 occurrences in Wikidata5M')

plt.show()


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

# %% Exploration: What are interesting aspects in the P21 relations extracted from the truthy triples file?

gender_facts_W5M_humans = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                   'data/interim/truthy_triples_file_for_W5M_gender_facts/all_gender_facts_W5M_human_entities_24032022.tsv'),
                                      sep = '\t',
                                      names = ['head_entity', 'relation', 'tail_entity'])

# How many humans in W5M have missing gender information? - 6886
number_of_missing_gender_facts = len(human_entities) - len(gender_facts_W5M_humans)

# Does anyone have more than one gender statement?
counts_heads = gender_facts_W5M_humans['head_entity'].value_counts()
gender_facts_W5M_humans['head_entity'].value_counts().unique()  # [4, 3, 2, 1]
counts_heads[counts_heads > 1]  # this is the case for 269 people
# 2 times 4 statements, 4 times 3 statements, the rest has 2 statements
# Most of these entities are queer people, e.g. non-binary --> multiple identities

# Do a quick plot of gender tail values:
counts_tails = gender_facts_W5M_humans['tail_entity'].value_counts()
counts_tails.plot.barh(logx = True,
                       title = 'Tail entity counts for gender facts extracted from truthy triples\n (filtered down to the 1.5M W5M human entities)')
plt.show()

# Save the distribution of tail entities for the W5M human entities
gender_facts_W5M_humans = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                   'data/interim/truthy_triples_file_for_W5M_gender_facts/all_gender_facts_W5M_human_entities_24032022.tsv'),
                                      sep = '\t',
                                      names = ['head_entity', 'relation', 'tail_entity'])

gender_tail_value_counts_filtered_to_W5M = gender_facts_W5M_humans['tail_entity'].value_counts()

# gender_tail_value_counts_filtered_to_W5M.to_csv(
#     'exploration/relationship_counts/only_W5M_humans_nttriples_dump_P21_tail_counts_24032022.csv')

