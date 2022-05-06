# %% Imports

import os

from src.utils import set_base_path_based_on_host, get_triples_df
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 50)
pd.set_option('display.max_rows', 50)

BASE_PATH_HOST = set_base_path_based_on_host()

# %% Step 1: Load the raw triples (uses the raw 168MB all_triplets file, not the splits)

# this accesses the all_triplets.txt file, not the transductive/inductive split
wikidata5m_raw_triples, _ = get_triples_df('wikidata5m')

# get information about size requirements
wikidata5m_raw_triples.info()  # size is 500 MB

# %% Step 2: Get list of all human entities contained in the original Wikidata5M file

# uses file: data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_all_triplets.txt
# creates file:  wikidata5m_human_entities_040122_v1.tsv

# filter for: <head entity>, is instance of (P31), human (Q5)
human_entities = wikidata5m_raw_triples['head_entity'][
    (wikidata5m_raw_triples['relation'] == 'P31') & (wikidata5m_raw_triples['tail_entity'] == 'Q5')]
human_entities.rename('human_head_entities', inplace = True)

# doublecheck that list of human entities only contains unique entries
human_entities_unique = human_entities.unique()
assert len(human_entities) == len(human_entities_unique), "There are duplicate human entities!"
human_entities = human_entities.reset_index(drop = True)

# SAVE TO TSV FOR MORE STABILITY THAN PICKLE!
# omit the column name and the index!
# human_entities.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data_preprocessing/wikidata5m_human_entities_040122_v1.tsv'),
#     sep = '\t', header = False, index = False)

# %% Step 3: Extract all facts from the original Wikidata5M that have a human head entity

# uses files:
#   data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_all_triplets.txt
#   wikidata5m_human_entities_040122_v1.tsv

# creates file: wikidata5m_human_facts_subset_complete_050122_v1.tsv

# require that the head entity is in the human entities list
human_triples_subset_mask = wikidata5m_raw_triples['head_entity'].isin(human_entities.values)
human_triples_subset = wikidata5m_raw_triples[human_triples_subset_mask]
# reset index
human_triples_subset = human_triples_subset.reset_index(drop = True)

# SAVE AS TSV
# omit the column name and the index!
# human_triples_subset.to_csv(os.path.join(BASE_PATH_HOST,
#                                          'data_preprocessing/wikidata5m_human_facts_subset_complete_050122_v1.tsv'),
#                             sep = '\t', header = False, index = False)

# %% Step 4: Extract all English Wikidata labels from the truthy triples dump

# idea: using labels is an alternative to using the Wikipedia descriptions provided by Wikidata5M
# --> shorter text = shorter model runtime (but also worse results)

# uses file:
#   latest-truthy_01012022.nt.gz
#   ntriples_extract_English_labels.sh

# creates file:
#   latest-truthy_01012022_English_labels.tsv

# 1. Download the truthy triples file (ca. 50GB) 'latest-truthy.nt.gz' from here: https://dumps.wikimedia.org/wikidatawiki/entities/
# 2. Make the shell script file executable (e.g. chmod) and run it: ./ntriples_extract_English_labels.sh
# 3. This results in a file where the first column are the Q-IDs, the second column the labels,
#    and the third column the language identifier, should be one of: en, en-ca, en-gb
# 3. Further process this in pandas to create the final file.

# %% Step 5: Merge multiple labels into one, creating one label per QID + PID

# load the ca. 86 million entity labels with all English labels
# there might be entities that do now have a "en" label, so we extract all of them and merge them

# uses file: latest-truthy_01012022_English_labels.tsv

# creates files:
#   entity2label_W5M_truthy_29032022_v1.tsv (4.9GB)
#   relation2label_W5M_truthy_29032022_v1.tsv (266KB)

# 86,244,142 labels overall
entity_labels_all = pd.read_csv(
    '/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/data/interim/truthy_triples_file_for_W5M_gender_facts/latest-truthy_01012022_English_labels.tsv',
    sep = '\t', names = ['ID', 'label', 'language'])

entity_labels_all.head()
entity_labels_all.info()  # 1.9 GB
entity_labels_all = entity_labels_all.convert_dtypes()

# doublecheck that only Enlish language labels are included
entity_labels_all['language'].value_counts()

entity_labels_all[entity_labels_all['language'] == 'de']
entity_labels_all[entity_labels_all['language'] == 'nl']
entity_labels_all[entity_labels_all['language'] == 'cs']
entity_labels_all[entity_labels_all['language'] == 'fr']
entity_labels_all[entity_labels_all['language'] == 'it']

entity_labels_all[entity_labels_all['language'].str.contains('pyridinone unit as a')]

### drop specific rows manually
# drop 4 rows where the label contains "@en"
# remove: 13333868, 13333870, 13333871
entity_labels_all[entity_labels_all['ID'] == 'Q29950078']  # also has English label
# remove: 64610527
entity_labels_all[entity_labels_all['ID'] == 'Q109730489']  # has no English label
# remove: 10382737, 10382738
entity_labels_all[entity_labels_all['ID'] == 'Q105986217']  # has no English label
# remove: 5299130
entity_labels_all[entity_labels_all['ID'] == 'Q56394716']  # also has English label
# remove: 42233701
entity_labels_all[entity_labels_all['ID'] == 'Q98510135']  # also has English label

# check the entity with the weird language, remove: 36366306
entity_labels_all[entity_labels_all['ID'] == 'Q43842663']

entity_labels_all = entity_labels_all.drop(
    index = [13333868, 13333870, 13333871, 64610527, 10382737, 10382738, 5299130, 42233701,
             36366306])

# reset the index
entity_labels_all.reset_index(drop = True, inplace = True)

# idea: work with sets of strings, this should make 1 string out of multiple ones
test = entity_labels_all.agg(['unique'])
# How many unique IDs?
len(test['ID'][0])  # 51129
# How many unique labels?
len(test['label'][0])  # 52289
# --> the labels are different

# determine IDs with only one occurrence to keep them, i.e. no conflicts
ID_value_counts = entity_labels_all['ID'].value_counts()
entities_with_one_label_index = ID_value_counts[ID_value_counts == 1].index
mask_entity_labels_with_one_label = entity_labels_all['ID'].isin(entities_with_one_label_index)
entity_labels_with_one_label = entity_labels_all[mask_entity_labels_with_one_label]  # ... rows
# drop the language column as it not needed anymore
entity_labels_with_one_label['language'].value_counts()
entity_labels_with_one_label = entity_labels_with_one_label.drop('language', axis = 1)

# For the remaining entities, extract a single label per Q-ID
entity_labels_with_more_than_one_label_index = ID_value_counts[ID_value_counts != 1].index
assert len(entity_labels_with_more_than_one_label_index) + len(
    entities_with_one_label_index) == len(entity_labels_all['ID'].unique())
mask_entity_labels_with_more_than_one_label = entity_labels_all['ID'].isin(
    entity_labels_with_more_than_one_label_index)
entity_labels_with_more_than_one_label = entity_labels_all[
    mask_entity_labels_with_more_than_one_label]

# look at the language counts
entity_labels_with_more_than_one_label['language'].value_counts()

# group the remaining dataframe by Q-IDs, create set of 'label' column
unique_labels = entity_labels_with_more_than_one_label.groupby('ID')['label'].apply(set)

# extract the rows where all labels are exactly the same, i.e. len(set) = 1
agreeing_labels = unique_labels[unique_labels.apply(len) == 1]
# transform labels from set to string
agreeing_labels = agreeing_labels.apply(lambda x: x.pop())
# make Q-ID index as column
agreeing_labels = agreeing_labels.reset_index()

# if they are different, keep the @en label
unique_labels = entity_labels_with_more_than_one_label.groupby('ID')['label'].apply(set)
not_agreeing_labels_index = unique_labels[unique_labels.apply(len) != 1].index
mask_not_agreeing_labels = entity_labels_with_more_than_one_label['ID'].isin(
    not_agreeing_labels_index)
entity_labels_not_agreeing = entity_labels_with_more_than_one_label[mask_not_agreeing_labels]

entity_labels_not_agreeing_only_en_label = entity_labels_not_agreeing[
    entity_labels_not_agreeing['language'] == 'en']
entity_labels_not_agreeing_only_en_label = entity_labels_not_agreeing_only_en_label.drop('language',
                                                                                         axis = 1)

# make sure that all the entities have an 'en' label
assert len(not_agreeing_labels_index) == len(entity_labels_not_agreeing_only_en_label)

# concatenate the final label file

entity_labels_final = pd.concat(
    [entity_labels_with_one_label, agreeing_labels, entity_labels_not_agreeing_only_en_label])

# assert that all Q-IDs are contained from the original import
assert set(entity_labels_all['ID'].unique()) == set(entity_labels_final['ID'].unique())

#  separate file into entity and relation labels
mask_entity_labels = entity_labels_final['ID'].str.startswith('Q')
entity_labels = entity_labels_final[mask_entity_labels]
mask_relation_labels = entity_labels_final['ID'].str.startswith('P')
relation_labels = entity_labels_final[mask_relation_labels]

assert len(entity_labels) + len(relation_labels) == len(entity_labels_final)

# save to TSV file
# (wikidata5m_wikipedia_texts.txt also is TSV between ID and text)
folder_to_save = os.path.join(BASE_PATH_HOST,
                              'data/interim/truthy_triples_file_for_W5M_gender_facts')

# entity_labels.to_csv(os.path.join(folder_to_save, 'entity2label_truthy_29032022_v1.tsv'),
#                      sep = '\t', header = False, index = False)
# relation_labels.to_csv(os.path.join(folder_to_save, 'relation2label_truthy_29032022_v1.tsv'),
#                        sep = '\t', header = False, index = False)

# %% Step 6: Extract a new human subset where all QIDs + PIDs have labels AND a description

# Caution: Because of changes made to Wikidata between 2019 (year of Wikidata5M creation)
# and 2022 (download of the truthy triples file), some Q/PIDs in Wikidata5M might not
# be contained in the truthy triples file.
# We will remove all facts containing such QIDs or PIDs.
# This will make the dataset slightly smaller, but is a necessary trade-off for being
# able to use the 2022 labels.

# uses files:
#   wikidata5m_human_facts_subset_complete_050122_v1.tsv
#   entity2label_W5M_truthy_29032022_v1
#   relation2label_W5M_truthy_29032022_v1.tsv
#   wikidata5m_wikipedia_texts.txt

# creates file: wikidata5m_human_facts_that_have_labels_and_descriptions_01042022_v2.tsv (184 MB)

human_triples_original_W5M = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                      'data/interim/wikidata5m_human_facts_subset_complete_050122_v1.tsv'),
                                         sep = '\t',
                                         names = ['head_entity', 'relation', 'tail_entity'])
all_entity_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                             'data/interim/truthy_triples_file_for_W5M_gender_facts/entity2label_W5M_truthy_29032022_v1.tsv'),
                                sep = '\t', names = ['ID', 'label'])
all_relation_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                               'data/interim/truthy_triples_file_for_W5M_gender_facts/relation2label_W5M_truthy_29032022_v1.tsv'),
                                  sep = '\t', names = ['ID', 'label'])
entity_descriptions = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                               'data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_wikipedia_texts.txt'),
                                  sep = '\t', names = ['ID', 'description'])

# create set of all QIDs appearing in the human facts subset
set_of_head_entities = set(human_triples_original_W5M['head_entity'])  # 1519261 IDs
set_of_tail_entities = set(human_triples_original_W5M['tail_entity'])  # 357775 IDs
set_of_all_QIDs_in_human_W5M = set_of_head_entities.union(set_of_tail_entities)
# In total there are 1750347 unique QIDs in the Wikidata5M subset.

# Which of these QIDs have a label (from the truthy file)?
human_subset_entity_labels_mask = all_entity_labels['ID'].isin(list(set_of_all_QIDs_in_human_W5M))
human_subset_entity_labels = all_entity_labels[human_subset_entity_labels_mask]
number_of_missing_labels = len(set_of_all_QIDs_in_human_W5M) - sum(human_subset_entity_labels_mask)
# 1743781 QIDs have an entity label, 6566 do not.
set_of_QIDs_with_truthy_label = set(human_subset_entity_labels['ID'])
set_of_QIDs_without_truthy_label = set_of_all_QIDs_in_human_W5M.difference(
    set_of_QIDs_with_truthy_label)

# Which of these QIDs have a description (from the original Wikidata5M file)?
human_subset_entity_descriptions_mask = entity_descriptions['ID'].isin(
    list(set_of_all_QIDs_in_human_W5M))
human_subset_entity_descriptions = entity_descriptions[human_subset_entity_descriptions_mask]
number_of_missing_descriptions = len(set_of_all_QIDs_in_human_W5M) - sum(
    human_subset_entity_descriptions_mask)
# 1739176 QIDs have a description, 11171 do not.
set_of_QIDs_with_description = set(human_subset_entity_descriptions['ID'])
set_of_QIDs_without_description = set_of_all_QIDs_in_human_W5M.difference(
    set_of_QIDs_with_description)

### Create the set of QIDs that have both a label and a description
set_of_QIDs_with_label_and_description = set_of_QIDs_with_description.intersection(
    set_of_QIDs_with_truthy_label)
set_of_QIDs_without_label_or_description = set_of_all_QIDs_in_human_W5M.difference(
    set_of_QIDs_with_label_and_description)
# 1732819 QIDs have an entity label AND a description.
number_of_QIDs_to_remove = len(set_of_all_QIDs_in_human_W5M) - len(
    set_of_QIDs_with_label_and_description)
# This means we exclude 17528 QIDs overall.

### Do the same for the PIDs but only for the truthy label (relations do not have descriptions)

set_of_all_PIDs = set(human_triples_original_W5M['relation'])
# In total there are 298 unique PIDs in the Wikidata5M subset.
human_subset_relation_labels_mask = all_relation_labels['ID'].isin(list(set_of_all_PIDs))
human_subset_relation_labels = all_relation_labels[human_subset_relation_labels_mask]
number_of_missing_relations = len(set_of_all_PIDs) - sum(human_subset_relation_labels_mask)
# 296 PIDs have an entity label, 2 do not.
set_of_PIDs_with_truthy_label = set(human_subset_relation_labels['ID'])  # 296 IDs
set_of_PIDs_without_truthy_label = set_of_all_PIDs.difference(set_of_PIDs_with_truthy_label)
# This means we exclude 2 PIDs.

### Exclude any facts from the human subset that contain a QID or PID that has *no* label or description

QIDs_without_label_or_description_list = list(set_of_QIDs_without_label_or_description)  # 17528 IDs
PIDs_without_truthy_label_list = list(set_of_PIDs_without_truthy_label)  # 2 IDs

human_triples_with_any_missing_label_or_description = human_triples_original_W5M.query(
    'head_entity == @QIDs_without_label_or_description_list | tail_entity == @QIDs_without_label_or_description_list | relation == @PIDs_without_truthy_label_list')
# This dataframe has 191178 rows that will be excluded.

# The remaining human subset contains 9613243 triples.
human_triples_that_have_labels_and_descriptions = human_triples_original_W5M.drop(
    index = human_triples_with_any_missing_label_or_description.index)

# save to TSV file
# human_triples_that_have_labels_and_descriptions.to_csv(os.path.join(BASE_PATH_HOST,
#                                                                     'data/interim/wikidata5m_human_facts_that_have_labels_and_descriptions_01042022_v2.tsv'),
#                                                        sep = '\t', header = False, index = False)


# %% Step 7: From this subset, extract the list of human entities

# Approach is the same as in Step 2

# uses file: wikidata5m_human_facts_that_have_labels_and_descriptions_01042022_v2.tsv

# creates file: wikidata5m_human_entities_01042022_v2.tsv (14MB)

# human_triples_that_have_labels_and_descriptions = pd.read_csv(os.path.join(BASE_PATH_HOST,
#                                          'data/interim/wikidata5m_human_facts_that_have_labels_and_descriptions_01042022_v2.tsv'),
#                             sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

human_triples_filtered_for_human_statements = human_triples_that_have_labels_and_descriptions[
    (human_triples_that_have_labels_and_descriptions['relation'] == 'P31') & (
            human_triples_that_have_labels_and_descriptions['tail_entity'] == 'Q5')]
human_triples_filtered_for_human_statements.head()

human_entities_new = human_triples_filtered_for_human_statements['head_entity']
# There are 1503491 human entities contained in the new subset.
# That means there are 15770 less than in the first human entities file.

# save to TSV
# omit the column name and the index!
# human_entities_new.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5m_human_entities_01042022_v2.tsv'),
#     sep = '\t', header = False, index = False)


# %% Step 8: Retrieve gender relations from a current truthy triples dump

# Use a shell script to retrieve the gender facts from a current Wikidata dump

# uses files:
#   latest-truthy_01012022.nt.gz
#   ntriples_extraction.sh
#   triple_ext_allP21.sh

# creates files:
#   latest-truthy_01012022_only_triples.txt
#   all_gender_facts_23032022.txt (174MB)

# 1. Change the file names inside 'ntriples_extraction.sh' according to your wishes
# 2. Make the file executable (e.g. chmod) and run it: ./ntriples_extraction.sh
# 3. You now have a clean 3 column file consisting of Q and PIDs (ca. 45GB).
# 4. Do 2.+ 3. for the script 'triple_ext_allP21.sh' as well.
# 5. You now have a file that contains all gender (P21) facts of the dump you downloaded.

# %% Step 9: For the remaining human entities, extract the P21 gender relations from the truthy triples file

# uses files:
#   all_gender_facts_23032022.txt
#   wikidata5m_human_entities_01042022_v2.tsv

# creates file: all_gender_facts_for_W5M_human_entities_01042022.tsv (32 MB)

all_gender_facts_dump = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                 'data/interim/truthy_triples_file_for_W5M_gender_facts/all_gender_facts_23032022.txt'),
                                    delim_whitespace = True,
                                    names = ['head_entity', 'relation', 'tail_entity'])
# doublecheck correct import
all_gender_facts_dump.head()
# there are 7,612,636 facts overall
all_gender_facts_dump.info()

# doublecheck that relation really is just P21
all_gender_facts_dump['relation'].unique()  # Yes!

# What are the tail values and their counts?
gender_tail_entities_count = all_gender_facts_dump['tail_entity'].value_counts()
# Save this to exploration/relationship_counts/
gender_tail_entities_count.to_csv(
    'exploration/relationship_counts/nttriples_dump_P21_tail_counts_01042022.csv')

# Load the list of human entities contained in W5M as a pd.Series (without column name)
human_entities = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5m_human_entities_01042022_v2.tsv'),
    header = None).squeeze('columns')
# doublecheck the import: Series is not supposed to have a column name!
human_entities.head()

gender_facts_W5M_humans_mask = all_gender_facts_dump['head_entity'].isin(human_entities.values)
gender_facts_W5M_humans = all_gender_facts_dump[gender_facts_W5M_humans_mask]

# So what does the result look like?
gender_facts_W5M_humans.head()
gender_facts_W5M_humans.info()

# save to TSV (32 MB)
# gender_facts_W5M_humans.to_csv(os.path.join(BASE_PATH_HOST,
#                                             'data/interim/truthy_triples_file_for_W5M_gender_facts/all_gender_facts_for_W5M_human_entities_01042022.tsv'),
#                                sep = '\t', header = False, index = False)

# %% Step 10: Add these gender facts to the human facts subset

# decision: Exploration showed that all tail entities except for "male, female" have
# counts less than 500 in gender_facts_W5M_humans, so we will not include them.

# uses files:
#   wikidata5m_human_facts_that_have_labels_and_descriptions_01042022_v2.tsv
#   all_gender_facts_for_W5M_human_entities_01042022.tsv

# creates file: wikidata5m_human_facts_with_gender_added_01042022_v3.tsv (216 MB)

# load the subset of human facts created above (9,613,243 triples)
human_triples_v2 = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                            'data/interim/wikidata5m_human_facts_that_have_labels_and_descriptions_01042022_v2.tsv'),
                               sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

# load all of my gender facts
gender_facts_W5M_humans = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                   'data/interim/truthy_triples_file_for_W5M_gender_facts/all_gender_facts_for_W5M_human_entities_01042022.tsv'),
                                      sep = '\t',
                                      names = ['head_entity', 'relation', 'tail_entity'])

# remove gender facts from the original Wikidata5M file (384 rows)
W5M_gender_facts_to_remove = human_triples_v2[human_triples_v2['relation'] == 'P21']
human_triples = human_triples_v2.drop(index = W5M_gender_facts_to_remove.index)

# only keep binary gender for entities: Q6581097, Q6581072
gender_IDs_to_keep = ['Q6581097', 'Q6581072']
gender_facts_W5M_humans_binary = gender_facts_W5M_humans[
    gender_facts_W5M_humans['tail_entity'].isin(gender_IDs_to_keep)]

# doublecheck this
assert len(gender_facts_W5M_humans_binary['tail_entity'].unique()) == len(gender_IDs_to_keep)

# add binary facts to human subset
human_triples_with_new_gender = pd.concat([human_triples, gender_facts_W5M_humans_binary])

### doublecheck that the tail entities for gender are binary
# 1243734 male entities, 258204 female entities
human_triples_with_new_gender['tail_entity'][
    human_triples_with_new_gender['relation'] == 'P21'].value_counts()

### Doublecheck that this subset has the same human entities as wikidata5m_human_entities_01042022_v2.tsv

human_entities = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5m_human_entities_01042022_v2.tsv'),
    header = None).squeeze('columns')
human_entities_with_gender = human_triples_with_new_gender['head_entity'][
    (human_triples_with_new_gender['relation'] == 'P31') & (
            human_triples_with_new_gender['tail_entity'] == 'Q5')]
assert len(human_entities_with_gender) == len(human_entities)
assert sorted(human_entities_with_gender) == sorted(human_entities)
# Sanity check: Are the human entities the same as the unique set of head entities?
assert len(human_entities) == len(human_triples_v2['head_entity'].unique())
# Are there the same number of unique head entities before and after removing the P21 facts?
assert len(human_triples['head_entity'].unique()) == len(human_triples_v2['head_entity'].unique())
# Are the (sorted) Q IDs also exactly the same?
assert sorted(human_triples['head_entity'].unique()) == sorted(
    human_triples_v2['head_entity'].unique())

# save this as TSV (216 MB)
# human_triples_with_new_gender.to_csv(os.path.join(BASE_PATH_HOST,
#                                                   'data/interim/wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv'),
#                                      sep = '\t', header = False, index = False)


# %% Step 11: Make sure that I have labels and descriptions for all IDs now

# uses files:
#   wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv
#   entity2label_truthy_29032022_v1
#   relation2label_truthy_29032022_v1.tsv
#   wikidata5m_wikipedia_texts.txt

human_triples_with_new_gender = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                         'data/interim/wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv'),
                                            sep = '\t',
                                            names = ['head_entity', 'relation', 'tail_entity'])
all_entity_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                             'data/interim/truthy_triples_file_for_W5M_gender_facts/entity2label_truthy_29032022_v1.tsv'),
                                sep = '\t', names = ['ID', 'label'])
all_relation_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                               'data/interim/truthy_triples_file_for_W5M_gender_facts/relation2label_truthy_29032022_v1.tsv'),
                                  sep = '\t', names = ['ID', 'label'])
all_entity_descriptions = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                   'data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_wikipedia_texts.txt'),
                                      sep = '\t', names = ['ID', 'description'])

set_of_head_entities = set(human_triples_with_new_gender['head_entity'])
set_of_tail_entities = set(human_triples_with_new_gender['tail_entity'])
set_of_human_entities = set_of_head_entities.union(set_of_tail_entities)
set_of_human_relations = set(human_triples_with_new_gender['relation'])
set_of_human_entities = sorted(set_of_human_entities)
set_of_human_relations = sorted(set_of_human_relations)

### Do I have a label for each of all the QIDs in the human subset with gender?
entity_labels_for_my_W5M_subset_mask = all_entity_labels['ID'].isin(set_of_human_entities)
entity_labels_for_my_W5M_subset = all_entity_labels[entity_labels_for_my_W5M_subset_mask]

# I expect that I get a label from the large file for each QID in my subset.
assert len(entity_labels_for_my_W5M_subset) == len(set_of_human_entities)
number_of_entity_labels_that_are_not_needed = len(all_entity_labels) - len(
    entity_labels_for_my_W5M_subset)
# 80,053,430 entity labels are not needed.

# check whether entity male - Q6581097 and female - Q6581072 have labels
entity_labels_for_my_W5M_subset[entity_labels_for_my_W5M_subset['ID'] == 'Q6581097']  # male
entity_labels_for_my_W5M_subset[entity_labels_for_my_W5M_subset['ID'] == 'Q6581072']  # female

### Do the same for the relation PIDs
relation_labels_for_my_W5M_subset_mask = all_relation_labels['ID'].isin(set_of_human_relations)
relation_labels_for_my_W5M_subset = all_relation_labels[relation_labels_for_my_W5M_subset_mask]

# I expect that I get a label from the large file for each QID in my subset.
assert len(relation_labels_for_my_W5M_subset) == len(set_of_human_relations)
number_of_relation_labels_that_are_not_needed = len(all_relation_labels) - len(
    relation_labels_for_my_W5M_subset)
# 9209 entity labels are not needed.
relation_labels_for_my_W5M_subset[relation_labels_for_my_W5M_subset['ID'] == 'P21']  # sex or gender

### Do the same for the entity QID descriptions (original Wikidata5M file)
entity_descriptions_for_my_W5M_subset_mask = all_entity_descriptions['ID'].isin(
    set_of_human_entities)
entity_descriptions_for_my_W5M_subset = all_entity_descriptions[
    entity_descriptions_for_my_W5M_subset_mask]

# I expect that I get a label from the large file for each QID in my subset.
# IMPORTANT: I do not, as I added the two entities for male + female!
assert len(set_of_human_entities) - len(entity_descriptions_for_my_W5M_subset) == 2
number_of_descriptions_labels_that_are_not_needed = len(all_entity_descriptions) - len(
    entity_descriptions_for_my_W5M_subset)
# 3,083,244 entity descriptions are not needed.

# check whether entity male - Q6581097 and female - Q6581072 have descriptions
# They don't have a description!
entity_descriptions_for_my_W5M_subset[entity_descriptions_for_my_W5M_subset['ID'] == 'Q6581097']
entity_descriptions_for_my_W5M_subset[entity_descriptions_for_my_W5M_subset['ID'] == 'Q6581072']

# %% TODO Step 12: Retrieve Wikipedia descriptions for the gender IDs I added

# uses files:
#   wikidata5m_wikipedia_texts.txt

# creates file:
#   wikidata5m_human_descriptions.tsv

all_entity_descriptions = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                   'data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_text.txt'),
                                      sep = '\t')
all_entity_descriptions.head()
# Q7594088
all_entity_descriptions.iloc[3]
# the text is the first part in the article before the table of contents

### Description of approach from the paper that created Wikidata5M
# Wang et al. (2021): KEPLER: A unified model for knowledge embedding and pre-trained language representation
# Chapter 3.1 Data Collection: "We use the July 2019 dump of Wikidata and
# Wikipedia. For each entity in Wikidata, we align it to its Wikipedia page and extract the
# first section as its description. Entities with no pages or with descriptions fewer than
# 5 words are discarded."

# IDs to extract: Q6581097, Q6581072

# Q6581097 (male) [from Wikipedia](https://en.wikipedia.org/wiki/Male)
# Q6581072 (female) [from Wikipedia](https://en.wikipedia.org/wiki/Female)

# added this to the original descriptions file using Linux' `cat`

# %% Step 13: Clean the original W5M descriptions file

# IMPORTANT: the solution!
# 1. Get the chronological list of QIds from the decriptions file: wikidata5m_wikipedia_texts_with_gender.txt
# 2. remove it from the file
# 3. replace all tab operators with whitespaces
# 4. add the QIDs back in again

# IMPORTANT problem descriptions: a decent number of strings are cut off at \t and
# therefore too short!

# original file has 4815262 entries + 2 gender IDs
fresh_descriptions = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                              "data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_wikipedia_texts_with_gender.txt"),
                                 sep = '\t', usecols = [0, 1], names = ['ID', 'description'])

# There are 1732021 unique QIDs in total for which I need descriptions!
with open(os.path.join(BASE_PATH_HOST, 'data/interim/KG_and_LM_files/clean_filenames/entities.txt'),
          'r') as f:
    HumanWikidata5M_all_QIDs = list(f.read().splitlines())
f.close()

my_subset_of_entity_descriptions_mask = fresh_descriptions['ID'].isin(HumanWikidata5M_all_QIDs)
my_subset_of_entity_descriptions = fresh_descriptions[my_subset_of_entity_descriptions_mask]

# I expect that I get a description from the large file for each QID in my subset.
assert len(my_subset_of_entity_descriptions) == len(HumanWikidata5M_all_QIDs)

# Look at my one problem case, Q3257645
my_subset_of_entity_descriptions[my_subset_of_entity_descriptions['ID'] == 'Q3257645']
# description = "Lobsang Wangyal (" so it was cut off!!!

# check whether the length is reasonable like this
my_subset_of_entity_descriptions['string_len'] = my_subset_of_entity_descriptions[
    'description'].apply(len)
# plot the distribution of lengths

# IMPORTANT: SOLUTION STARTS HERE!

description_path = os.path.join(BASE_PATH_HOST,
                                "data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_text.txt")

save_path = os.path.join(BASE_PATH_HOST, 'data/interim/clean_descriptions')

# IMPORTANT: Do step 1 for the original file WITHOUT the 2  gender descriptions
list_of_entities = []
with open(description_path, 'r') as f:
    ent_lines = f.readlines()
for line in ent_lines:
    temp = line.strip().split('\t')
    list_of_entities.append(temp[0])
# This results in all 4815483 entities (without gender)

# save to path
with open(os.path.join(save_path, 'entities_ordered_original_W5M_22042022.txt'), 'w') as f:
    f.writelines(''.join([str(x) + '\n' for x in list_of_entities]))
f.close()


################### AFTER CHANGING THE FILES

# IMPORTANT: try whether it works with the code that loads
# it in the KG-BERT script - yay it is better now!
# this is the original code for the FB15K-237 long descriptions
ent2text_all_first_two = {}
with open(os.path.join(save_path, "tryit.txt"), 'r') as f:
    ent_lines = f.readlines()
for line in ent_lines:
    temp = line.strip().split('\t')
    # first_sent_end_position = temp[1].find(".")
    ent2text_all_first_two[temp[0]] = temp[1]  # [:first_sentence_end_position + 1]

assert len(ent2text_all_first_two) == len(list_of_entities)

# try whether it worked with pandas
# STILL ONLY 4815265
changed_descriptions = pd.read_csv(os.path.join(save_path,
                                              "wikidata5m_cleaned_text_with_gender_22042022_v2.txt"),
                                 sep = '\t', names = ['ID', 'description'])

my_subset_of_entity_descriptions_mask = changed_descriptions['ID'].isin(HumanWikidata5M_all_QIDs)
my_subset_of_entity_descriptions = changed_descriptions[my_subset_of_entity_descriptions_mask]

# I expect that I get a description from the large file for each QID in my subset.
assert len(my_subset_of_entity_descriptions) == len(HumanWikidata5M_all_QIDs)

my_subset_of_entity_descriptions.to_csv(os.path.join(save_path, 'HumanWikidata5M_descriptions_22042022_v1.tsv'),
                                        sep = '\t', header = False, index = False)

my_subset_of_entity_descriptions['string_len'] = my_subset_of_entity_descriptions[
    'description'].apply(len)
my_subset_of_entity_descriptions.query('string_len<50')

my_subset_of_entity_descriptions.to_csv(os.path.join(save_path, 'HumanWikidata5M_clean_descriptions_for_easy_import_with_length_counts.tsv'),
                                        sep = '\t')





# %% Step 14: Create files needed for KG+LM models + save filtered entity2label and relation2label files

# uses files:
#   wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv
#   entity2label_truthy_29032022_v1
#   relation2label_truthy_29032022_v1.tsv
#   wikidata5m_wikipedia_texts.txt

# creates files
#   data/interim/KG_and_LM_files/entities_04042022_v1.txt (14.9MB)
#   data/interim/KG_and_LM_files/relations_04042022_v1.txt (1.5KB)
#   entity2label_wikidata5m_human_08042022_v2.tsv (43MB)
#   relation2label_wikidata5m_human_08042022_v2.tsv (5.8KB)

human_triples_with_new_gender = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                         'data/interim/wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv'),
                                            sep = '\t',
                                            names = ['head_entity', 'relation', 'tail_entity'])
all_entity_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                             'data/interim/truthy_triples_file_for_W5M_gender_facts/entity2label_truthy_29032022_v1.tsv'),
                                sep = '\t', names = ['ID', 'label'])
all_relation_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                               'data/interim/truthy_triples_file_for_W5M_gender_facts/relation2label_truthy_29032022_v1.tsv'),
                                  sep = '\t', names = ['ID', 'label'])
all_entity_descriptions = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                   'data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_wikipedia_texts.txt'),
                                      sep = '\t', names = ['ID', 'description'])
set_of_head_entities = set(human_triples_with_new_gender['head_entity'])
set_of_tail_entities = set(human_triples_with_new_gender['tail_entity'])
set_of_human_entities = set_of_head_entities.union(set_of_tail_entities)
set_of_human_relations = set(human_triples_with_new_gender['relation'])

# sort all IDs alphanumerically! - more clarity when used in different files
set_of_human_entities = sorted(set_of_human_entities)
set_of_human_relations = sorted(set_of_human_relations)

folder_to_save_KG_and_LM = os.path.join(BASE_PATH_HOST, 'data/interim/KG_and_LM_files')

### entities.txt: list of all entities in the dataset
with open(os.path.join(folder_to_save_KG_and_LM, 'entities_04042022_v1.txt'), 'w') as f:
    f.writelines(''.join([str(x) + '\n' for x in set_of_human_entities]))
f.close()

### relations.txt: list of all relations in the dataset
with open(os.path.join(folder_to_save_KG_and_LM, 'relations_04042022_v1.txt'), 'w') as f:
    f.writelines(''.join([str(x) + '\n' for x in set_of_human_relations]))
f.close()

### Filter the entity2text and relation2text files created in step 5
entity_labels_for_my_W5M_subset_mask = all_entity_labels['ID'].isin(set_of_human_entities)
entity_labels_for_my_W5M_subset = all_entity_labels[entity_labels_for_my_W5M_subset_mask]
len(entity_labels_for_my_W5M_subset)  # 1732021 labels

# do the same for the relation labels
relation_labels_for_my_W5M_subset_mask = all_relation_labels['ID'].isin(set_of_human_relations)
relation_labels_for_my_W5M_subset = all_relation_labels[relation_labels_for_my_W5M_subset_mask]
len(relation_labels_for_my_W5M_subset)  # 292 labels

# sort IDs alphanumerically
entity_labels_for_my_W5M_subset_sorted = entity_labels_for_my_W5M_subset.sort_values(by = 'ID',
                                                                                     ascending = True)
assert list(entity_labels_for_my_W5M_subset_sorted['ID']) == list(set_of_human_entities)
relation_labels_for_my_W5M_subset_sorted = relation_labels_for_my_W5M_subset.sort_values(by = 'ID',
                                                                                         ascending = True)
assert list(relation_labels_for_my_W5M_subset_sorted['ID']) == list(set_of_human_relations)

# Save new entity2label (43MB) and relation2label (5.8 KB) files
# entity_labels_for_my_W5M_subset_sorted.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data/interim/entity2label_wikidata5m_human_08042022_v2.tsv'),
#     sep = '\t', header = False, index = False)
# relation_labels_for_my_W5M_subset_sorted.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data/interim/relation2label_wikidata5m_human_08042022_v2.tsv'),
#     sep = '\t', header = False, index = False)

### entity2label.tsv: 1 column QIDs, one column label, i.e. short text version
# simple copy the file created in the previous step to folder KG_and_LM_files

### relation2label.tsv: 1 column PIDs, one column label, i.e. short text version
# simple copy the file created in the previous step to folder KG_and_LM_files

### TODO entity2textlong.tsv: filter down the original Wikidata5M description file
entity_descriptions = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                               'data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_text.txt'),
                                  sep = '\t', names = ['ID', 'description'])
# doublecheck the import
entity_descriptions.head()
entity_descriptions.info()

# only keep the QID descriptions that are in my human facts subset
my_subset_of_entity_descriptions_mask = entity_descriptions['ID'].isin(set_of_human_entities)
my_subset_of_entity_descriptions = entity_descriptions[my_subset_of_entity_descriptions_mask]

# I expect that I get a description from the large file for each QID in my subset.
assert len(my_subset_of_entity_descriptions) == len(set_of_human_entities)

entity_descriptions_filtered_sorted = my_subset_of_entity_descriptions.sort_values(by = 'ID',
                                                                                   ascending = True,
                                                                                   ignore_index = True)

### TODO train.tsv, dev.tsv, test.tsv --> decide for a split

from src.utils_with_pykeen import create_train_val_test_split_from_single_TSV

create_train_val_test_split_from_single_TSV(train_val_test_split = (0.98, 0.01, 0.01),
    rel_path_to_human_facts_file= 'data/interim/wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv')

# Keidar: did an 0.8 0.1 0.1 split
# with about 9 million human facts, this results in 900,000 valid + test triples

# KEPLER paper: in transductive split, the validation + test set is "tiny"
# valid: only 5163 triples of 20624575 triples (0.025%)
# test: only 5133 triples of 20624575 triples (0.02489%)

test_data_dot01 = pd.read_csv(os.path.join(BASE_PATH_HOST, 'data/interim/test_data_0.01_rs42_05_05_2022_10:29.tsv'),
                              sep = '\t', names = ['head', 'relation', 'tail'])

test_data_dot01[test_data_dot01['relation'] == 'P21']  # 14539
test_data_dot01[test_data_dot01['relation'] == 'P106']  # 4889

test_data_dot1 = pd.read_csv(os.path.join(BASE_PATH_HOST, 'data/interim/test_data_0.1_rs42_06_04_2022_18:10.tsv'),
                              sep = '\t', names = ['head', 'relation', 'tail'])

test_data_dot1[test_data_dot1['relation'] == 'P21']  # 146035
test_data_dot1[test_data_dot1['relation'] == 'P106']  # 49515


# %% Step 15: Create utility file for KG only: map from numeric ID to string ID to text label

# first column: entity/relation ID
# second column: numeric ID starting at 0
# third column: English label (extracted from truthy file)
# IMPORTANT: this order matters for utils.HumanWikidata5M_pykeen._load!

# uses files:
#   entity2label_W5M_truthy_06042022_v2.tsv
#   relation2label_W5M_truthy_06042022_v2.tsv

# creates files:
#   entity_ID_to_numID_to_label_08042022_v1.tsv
#   relation_ID_to_numID_to_label_08042022_v1.tsv

entity_labels = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/entity2label_wikidata5m_human_08042022_v2.tsv'),
    sep = '\t', names = ['ID', 'label'])
relation_labels = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/relation2label_wikidata5m_human_08042022_v2.tsv'),
    sep = '\t', names = ['ID', 'label'])

# add index as numeric column
entity_labels['numeric_relation_ID'] = entity_labels.index
relation_labels['numeric_relation_ID'] = relation_labels.index

# reorder the columns as needed
entity_ID_to_numID_to_label = entity_labels[['ID', 'numeric_relation_ID', 'label']]
relation_ID_to_numID_to_label = relation_labels[['ID', 'numeric_relation_ID', 'label']]

# Save to TSV
# entity_ID_to_numID_to_label.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data/interim/KG_only_files/entity_ID_to_numID_to_label_08042022_v1.tsv'),
#     sep = '\t', header = False, index = False)
# relation_ID_to_numID_to_label.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data/interim/KG_only_files/relation_ID_to_numID_to_label_08042022_v1.tsv'),
#     sep = '\t', header = False, index = False)


# %% TODO handle the Unicode literals in the label file

# 7/4/ try out importing it first with Python's IOTextWrapper

# uses file:
#   old file: entity2label_wikidata5m_human_08042022_v2.tsv

# creates file:

relation_labels = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/relation2label_wikidata5m_human_08042022_v2.tsv'),
    sep = '\t', names = ['ID', 'label'])

entity_labels = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/entity2label_wikidata5m_human_08042022_v2.tsv'),
    sep = '\t', names = ['ID', 'label'])


# Manojs code snippet
def replace_unicode(item):
    res_item = item.encode("utf-8").decode()
    print(item, " - ", res_item)
    return res_item


for label in entity_labels.head()['label']:
    replace_unicode(label)

with open('data/interim/old/entity2label_W5M_truthy_31032022_v2.tsv', 'r', encoding = 'utf-8') as f:
    all_lines = f.read().splitlines()
    for line in all_lines:
        replace_unicode(line)

with open('filename.txt', 'r', encoding = 'utf-8') as f:
    entity_to_label = []
    all_lines = f.read().splitlines()
    for line in all_lines:
        line = replace_unicode(line)
        one_line = line.strip().split('\t')
        entity_to_label.append(one_line)
entity_to_label

with open('data/interim/old/entity2label_W5M_truthy_31032022_v2.tsv', encoding = 'utf-8') as f:
    for line in f:
        print(repr(line))

# generally useful: https://docs.python.org/3/howto/unicode.html#unicode-literals-in-python-source-code

# Try an example
# useless source for python 2: https://stackoverflow.com/questions/21646245/how-to-decode-a-text-with-unicodes-like-u00e7-in-python
test_string_raw = r'Wilhelm D\u00F6rpfeld'
print(test_string_raw)
# versus
test_string = 'Wilhelm D\u00F6rpfeld'
print(test_string)  # special character is recognized by Python



# %% After all processing of the full HumanWikidata5M dataset: create a subset

# decision made on 6th May that training the models with the full 11 million triples
# is too time-consuming, so we will use a subset of the HumanWikidata5M dataset

# load the v3 version of HumanWikidata5M
full_HumanWikidata5M = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                'data/interim/wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv'),
                                   sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

assert len(full_HumanWikidata5M) == 11114797  # size of HumanWikidata5M after adding gender

random_seed = 42

# I decided to make it 10 times larger than FB15K-237.
# 10 * 310,116 = 3,101,160
target_dataset_size = 3101160

# create new dataframe with new index 0,1,...,n-1
subset_HumanWikidata5M = full_HumanWikidata5M.sample(
    n = target_dataset_size, replace = False, random_state = random_seed,
    ignore_index = True, axis = 0  # sample rows = 0
)

# # save subset as v4 to disk /61MB)
# subset_HumanWikidata5M.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5M_human_facts_subset_060522_v4.tsv'),
#     sep = '\t', header = False, index = False)

# Regarding the split, I decided for (0.9, 0.05, 0.05)

from utils_with_pykeen import create_train_val_test_split_from_single_TSV

# create split and directly save it to data/interim
create_train_val_test_split_from_single_TSV(train_val_test_split = (0.9, 0.05, 0.05),
                                            rel_path_to_human_facts_file = 'data/interim/wikidata5M_human_facts_subset_060522_v4.tsv',
                                            random_state = random_seed)





