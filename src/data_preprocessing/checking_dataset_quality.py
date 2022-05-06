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

# %% Exploration: Can I use the alias files from the original Wikidata5M files?

# Idea: Can I use the first alias as the label for an entity/relation?
# Problem: There are data quality issues for the entity files! i.e. typos, etc.
# I can't use these files, and will instead extract the labels from a recent dump!

# for the labels, only access the first two columns
# the columns beyond these are the aliases, not the label
entity_labels_raw = pd.read_csv('./data/interim/wikidata5m_entity_aliases.txt', sep = '\t',
                                usecols = [0, 1], names = ['wikidata_ID', 'label'])
# correct import: provide names, only use first two columns
relation_labels_raw = pd.read_csv('./data/interim/wikidata5m_relation_aliases.txt', sep = '\t',
                                  usecols = [0, 1], names = ['wikidata_ID', 'label'])

# sort the dataframe according to the Wikidata IDs in ascending order
# (this makes it more reproducible)
entity_labels_sorted = entity_labels_raw.sort_values(by = 'wikidata_ID', ascending = True,
                                                     ignore_index = True)
relation_labels_sorted = relation_labels_raw.sort_values(by = 'wikidata_ID', ascending = True,
                                                         ignore_index = True)

# add index as numeric column
# (numeric ID mapping will later be used by pykeen
entity_labels_sorted['numeric_relation_ID'] = entity_labels_sorted.index
relation_labels_sorted['numeric_relation_ID'] = relation_labels_sorted.index

# save this dataframe to disk, always use it!
# save first column as entities.txt and relations.txt
# don't write index and column name
entity_labels_sorted['wikidata_ID'].to_csv('./data/interim/KG_and_LM/entities.txt', index = False,
                                           header = False)
relation_labels_sorted['wikidata_ID'].to_csv('./data/interim/KG_and_LM/relations.txt',
                                             index = False, header = False)

# idea: save 1st and 2nd column as entity2text.txt

my_human_entities = pd.read_csv('data/interim/wikidata5m_human_entities_040122.tsv', sep = '\t')
my_human_facts = pd.read_csv('data/interim/wikidata5m_human_facts_subset_complete_050122.tsv',
                             sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

len(set(my_human_facts.iloc[:, 1]))
human_relations = set(my_human_facts.iloc[:, 1])  # 298

# save dataframes as CSV: entities_NumID_WikidataID_label.csv, relations_NumID_WikidataID_label.csv

# %% Exploration: Do the relation labels of Wikidata5M and truthy triples largely agree?

# load the original Wikidata5M relation alias file
relation_labels_original_W5M = pd.read_csv('./data/interim/wikidata5m_relation_aliases.txt',
                                           sep = '\t', usecols = [0, 1],
                                           names = ['wikidata_ID', 'label'])

# load my version of this value based on truthy triples file
relation_labels_truthy = pd.read_csv(
    './data/interim/truthy_triples_file_for_W5M_gender_facts/relation2label_W5M_truthy_29032022_v1.tsv',
    sep = '\t', names = ['wikidata_ID', 'label'])

# W5M file only has 825 relations, the truthy triple file has 9501

# get the intersection of the PIDs
PIDs_original_W5M = set(relation_labels_original_W5M['wikidata_ID'])
PIDs_truthy = set(relation_labels_truthy['wikidata_ID'])

PIDs_in_both_files = PIDs_original_W5M.intersection(PIDs_truthy)
# these are 821 IDs --> 4 IDs are in W5M, but not in truthy
# Which ones? - 'P134', 'P1432', 'P1773', 'P2157'
PIDs_not_in_truthy = PIDs_original_W5M.difference(PIDs_in_both_files)

relation_labels_original_W5M[relation_labels_original_W5M['wikidata_ID'] == 'P134']
relation_labels_original_W5M[relation_labels_original_W5M['wikidata_ID'] == 'P1432']
relation_labels_original_W5M[relation_labels_original_W5M['wikidata_ID'] == 'P1773']
relation_labels_original_W5M[relation_labels_original_W5M['wikidata_ID'] == 'P2157']

### Do the labels on the intersection PIDs agree in all cases? - No.

# select the common PIDs for one dataframe
intersection_labels_original_W5M = relation_labels_original_W5M[
    relation_labels_original_W5M['wikidata_ID'].isin(list(PIDs_in_both_files))].reset_index(
    drop = True)
# and the other
intersection_labels_truthy = relation_labels_truthy[
    relation_labels_truthy['wikidata_ID'].isin(list(PIDs_in_both_files))].reset_index(drop = True)

intersection_labels_original_W5M.sort_values(by = 'wikidata_ID', ascending = True,
                                             ignore_index = True, inplace = True)
intersection_labels_truthy.sort_values(by = 'wikidata_ID', ascending = True, ignore_index = True,
                                       inplace = True)

compare_both_labels_df = intersection_labels_original_W5M.rename(
    columns = {'label': 'label_original_W5M'})
compare_both_labels_df['label_truthy_file'] = intersection_labels_truthy['label']

# use apply to add  True/False column
compare_both_labels_df['agreement'] = compare_both_labels_df['label_original_W5M'] == \
                                      compare_both_labels_df['label_truthy_file']

len(compare_both_labels_df['agreement']) - sum(compare_both_labels_df['agreement'])
# only 39 labels do not agree
# Which ones are those?
disagreeing_labels = compare_both_labels_df[compare_both_labels_df['agreement'] == False]

### Do the 4 missing labels (of the truthy file) appear anywhere in the human subset?
human_triples = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5m_human_facts_subset_complete_050122.tsv'),
    sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

set_of_human_subset_PIDs = set(human_triples['relation'].unique())
# overall 298 relations
for item in PIDs_not_in_truthy:
    is_in_human_subset = item in set_of_human_subset_PIDs
    print(
        f'Relation {item} is contained in the set of PIDs appearing in my human facts subset: {is_in_human_subset}')  # No ,they don't! So it's not relevant for my use case
