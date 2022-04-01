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

# %% Step 1: Load the raw triples

# this accesses the all_triplets.txt file, not the transductive/inductive split
wikidata5m_raw_triples, _ = get_triples_df('wikidata5m')

# get information about size requirements
wikidata5m_raw_triples.info()  # size is 500 MB

# %% Step 2: Get list of all human entities

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
#     os.path.join(BASE_PATH_HOST, 'data_preprocessing/wikidata5m_human_entities_040122.tsv'),
#     sep = '\t', header = False, index = False)

# %% Step 3: Filter triples_df down to only the triples that have a human head entity

# require that the head entity is in the human entities list
human_triples_subset_mask = wikidata5m_raw_triples['head_entity'].isin(human_entities.values)
human_triples_subset = wikidata5m_raw_triples[human_triples_subset_mask]
# reset index
human_triples_subset = human_triples_subset.reset_index(drop = True)

# SAVE AS TSV
# omit the column name and the index!
# human_triples_subset.to_csv(os.path.join(BASE_PATH_HOST,
#                                          'data_preprocessing/wikidata5m_human_facts_subset_complete_050122.tsv'),
#                             sep = '\t', header = False, index = False)

# %% Step 4: Retrieve gender relations from a current truthy triples dump

# Use a shell script to retrieve the gender facts from a current Wikidata dump

# shell script folder: src/data_preprocessing/extract_gender_facts_from_ntdump
# data folder: data/interim/truthy_triples_file_for_W5M_gender_facts

# 1. Download the truthy triples file (ca. 50GB) 'latest-truthy.nt.gz' from here: https://dumps.wikimedia.org/wikidatawiki/entities/
# 2. Change the file names inside 'ntriples_extraction.sh' according to your wishes
# 3. Make the file executable (e.g. chmod) and run it: ./ntriples_extraction.sh
# 4. You now have a clean 3 column file consisting of Q and PIDs (ca. 45GB).
# 5. Do 2.+ 3. for the script 'triple_ext_allP21.sh' as well.
# 6. You now have a file that contains all gender (P21) facts of the dump you downloaded.

# resulting file: all_gender_facts_23032022.txt (174MB)

# %% Step 5: Only keep the gender facts for the human entities in Wikidata5M

all_gender_facts_dump = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                 'data/interim/truthy_triples_file_for_W5M_gender_facts/all_gender_facts_23032022.txt'),
                                    delim_whitespace = True,
                                    names = ['head_entity', 'relation', 'tail_entity'])
# doublecheck correct import
all_gender_facts_dump.head()
# there are 7,612,635 facts overall
all_gender_facts_dump.info()

# doublecheck that relation really is just P21
all_gender_facts_dump['relation'].unique()  # Yes!

# What are the tail values and their counts?
gender_tail_entities_count = all_gender_facts_dump['tail_entity'].value_counts()
# Save this to exploration/relationship_counts/
gender_tail_entities_count.to_csv(
    'exploration/relationship_counts/nttriples_dump_P21_tail_counts_24032022.csv')

# Load the list of human entities contained in W5M as a pd.Series (without column name)
human_entities = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5m_human_entities_040122.tsv'),
    header = None).squeeze('columns')
# doublecheck the import: Series is not supposed to have a column name!
human_entities.head()

# Now use the same approach as in Step 3:
human_triples_W5M_mask = all_gender_facts_dump['head_entity'].isin(human_entities.values)
gender_facts_W5M_humans = all_gender_facts_dump[human_triples_W5M_mask]
gender_facts_W5M_humans = gender_facts_W5M_humans.reset_index(drop = True)

# So what does the result look like?
gender_facts_W5M_humans.info()

# save to tsv (32 MB)
# gender_facts_W5M_humans.to_csv(os.path.join(BASE_PATH_HOST,
#                                             'data/interim/truthy_triples_file_for_W5M_gender_facts/all_gender_facts_W5M_human_entities_24032022.tsv'),
#     sep = '\t', header = False, index = False)


# %% TODO Step 6: Join the new gender facts with the human subset of W5M

# decision: Exploration showed that all tail entities except for "male, female" have
# counts less than 500 in gender_facts_W5M_humans, so we will not include them.

# load the subset of human facts created above (9,804,421 facts)
human_triples = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5m_human_facts_subset_complete_050122.tsv'),
    sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

# load all of my gender facts (1,512,374 facts)
gender_facts_W5M_humans = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                   'data/interim/truthy_triples_file_for_W5M_gender_facts/all_gender_facts_W5M_human_entities_24032022.tsv'),
                                      sep = '\t',
                                      names = ['head_entity', 'relation', 'tail_entity'])

# remove gender facts from the original Wikidata5M file (392 rows)
W5M_gender_facts_to_remove = human_triples[human_triples['relation'] == 'P21'].index
human_triples = human_triples.drop(index = W5M_gender_facts_to_remove)

# only keep binary gender for entities: Q6581097, Q6581072
gender_IDs_to_keep = ['Q6581097', 'Q6581072']
gender_facts_W5M_humans_binary = gender_facts_W5M_humans[
    gender_facts_W5M_humans['tail_entity'].isin(gender_IDs_to_keep)]

# doublecheck this
assert len(gender_facts_W5M_humans_binary['tail_entity'].unique()) == len(gender_IDs_to_keep)

# add binary facts to human subset
human_triples_with_new_gender = pd.concat([human_triples, gender_facts_W5M_humans_binary])

# doublecheck that the tail entities for gender are binary
human_triples_with_new_gender['tail_entity'][
    human_triples_with_new_gender['relation'] == 'P21'].value_counts()

# save this as tsv
# human_triples_with_new_gender.to_csv(os.path.join(BASE_PATH_HOST,
#                                                   'data/interim/wikidata5m_human_facts_with_gender_added_30032022.tsv'),
#                                      sep = '\t', header = False, index = False)

# %% Step 7: Extract the English Wikdiata labels from the truthy triples dump as well

# idea: using labels is an alternative to using the Wikipedia descriptions provided by Wikidata5M
# --> shorter text = shorter model runtime (but also worse results)

# script folder src/data_preprocessing/extract_gender_facts_from_ntdump

# for details refer to Step 4

# 1. Use the script ntriples_extract_English_labels.sh on the same truthy triples file.
# 2. This results in a file where the first column are the Q-IDs and the second
#    column the English labels, including Canadian/GB English
# 3. Further process this in pandas to create the final file.

# %% Step 8: Merge English labels into one

# load the ca. 86 million entity labels with all English labels
# @en, @en-gb, @en-ca,...
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
# entity_labels_with_one_label  # ... rows
# agreeing_labels  # ... rows
# entity_labels_not_agreeing_only_en_label  # ... rows

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

# entity_labels.to_csv(os.path.join(folder_to_save, 'entity2label_W5M_truthy_29032022_v1.tsv'),
#                      sep = '\t', header = False, index = False)
# relation_labels.to_csv(os.path.join(folder_to_save, 'relation2label_W5M_truthy_29032022_v1.tsv'),
#                        sep = '\t', header = False, index = False)


# %% Step 9: Remove triples that contain entities/relations without a truthy_triples label

# Why? - The labels are from 2022, but the Wikdiata5M triples are from the July 2019 dump.
# This means there will be some changes, i.e. Q/PIDs that were contained in Wikidata then,
# but aren't anymore.
# This will make the dataset slightly smaller, but is a necessary trade-off for being
# able to use the 2022 labels.

# IMPORTANT: Use the file created in step 6! (with newly added gender facts)
human_triples = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                         'data/interim/wikidata5m_human_facts_with_gender_added_30032022_v2.tsv'),
                            sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

all_entity_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                             'data/interim/truthy_triples_file_for_W5M_gender_facts/entity2label_W5M_truthy_29032022_v1.tsv'),
                                sep = '\t', names = ['ID', 'label'])
all_relation_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                               'data/interim/truthy_triples_file_for_W5M_gender_facts/relation2label_W5M_truthy_29032022_v1.tsv'),
                                  sep = '\t', names = ['ID', 'label'])

# create set of all QIDs appearing in the dataset
set_of_head_entities = set(human_triples['head_entity'])  # 1519261 IDs
set_of_tail_entities = set(human_triples['tail_entity'])  # 357768 IDs
set_of_all_QIDs = set_of_head_entities.union(set_of_tail_entities)  # 1750340 IDs
# In total there are 1750340 unique QIDs in the Wikidata5M subset.

### IMPORTANT: Determine whether there are any missing labels
human_subset_entity_labels_mask = all_entity_labels['ID'].isin(list(set_of_all_QIDs))
human_subset_entity_labels = all_entity_labels[human_subset_entity_labels_mask]

# Are there any QIDs for which I do not have a label? - Yes, for 6566 IDs...
assert len(set_of_all_QIDs) == len(human_subset_entity_labels)
number_of_missing_labels = len(set_of_all_QIDs) - sum(human_subset_entity_labels_mask)

# try the same with query
Q_IDs_to_keep = list(set_of_all_QIDs)
test_query = all_entity_labels.query('ID == @Q_IDs_to_keep')
# This also has only 1743774 rows!!! --> 6566 are missing..

# look at the entities which do not have a label
set_of_QIDs_with_truthy_label = set(test_query['ID'])  # 1743774 IDs
set_of_QIDs_without_truthy_label = set_of_all_QIDs.difference(set_of_QIDs_with_truthy_label)

# try again: is this label really not contained in all_entity_labels?
# select some QIDs randomly from set_of_QIDs_without_truthy_label and check them on wikidata.org

# 'Q22280068' - does not exist in current Wikidata
all_entity_labels[all_entity_labels['ID'] == 'Q22280068']
human_triples[human_triples['head_entity'] == 'Q22280068']
# doublecheck whether any of the Q-IDs appears in entity labels
set_of_truthy_entity_labels = set(all_entity_labels['ID'])
for item in set_of_QIDs_without_truthy_label:
    contained = item in set_of_truthy_entity_labels
    if contained:
        print(
            f'{item} is contained in truthy labels: {contained}')  # Nothing was printed, so they are not in there!

# How many facts do I remove, if I remove all triples where these 6566 entities are contained?
QIDs_without_truthy_label_list = list(set_of_QIDs_without_truthy_label)
# this dataframe contains 26,871 triples I need to remove
human_triples_with_missing_entity_label = human_triples.query(
    'head_entity == @QIDs_without_truthy_label_list | tail_entity == @QIDs_without_truthy_label_list')

# 20878 triples
head_missing = human_triples.query('head_entity == @QIDs_without_truthy_label_list')
# 6002 triples
tail_missing = human_triples.query('tail_entity == @QIDs_without_truthy_label_list')
# thereof 9 triples where both head + tail is missing


### IMPORTANT: Do the same missing labels analysis for the PIDs (relations)
set_of_all_PIDs = set(human_triples['relation'])
# In total there are 298 unique PIDs in the Wikidata5M subset.

human_subset_relation_labels_mask = all_relation_labels['ID'].isin(list(set_of_all_PIDs))
human_subset_relation_labels = all_relation_labels[human_subset_relation_labels_mask]

# Are there any PIDs for which I do not have a label? - Yes, for 2 IDs.
assert len(set_of_all_PIDs) == len(human_subset_relation_labels)
number_of_missing_relation_labels = len(set_of_all_PIDs) - sum(human_subset_relation_labels_mask)

# look at the relations which do not have a label
set_of_PIDs_with_truthy_label = set(human_subset_relation_labels['ID'])  # 296 IDs
set_of_PIDs_without_truthy_label = set_of_all_PIDs.difference(set_of_PIDs_with_truthy_label)

# manually look up these relations in wikidata.org: {'P1962', 'P2439'}
# P1962: was deleted on 24 August 2018
# P2439 was deleted on 4 June 2018

PIDs_without_truthy_label_list = list(set_of_PIDs_without_truthy_label)
# this dataframe contains 302 triples I need to remove
human_triples_with_missing_relation_label = human_triples.query(
    'relation == @PIDs_without_truthy_label_list')

### remove facts where I don't have a truthy label (either QID or PID)
# removes 27,173 triples
human_triples_with_any_missing_label = human_triples.query(
    'head_entity == @QIDs_without_truthy_label_list | tail_entity == @QIDs_without_truthy_label_list | relation == @PIDs_without_truthy_label_list')
human_triples_that_have_labels = human_triples.drop(
    index = human_triples_with_any_missing_label.index)

# human_triples_that_have_labels.to_csv(os.path.join(BASE_PATH_HOST,
#                                                    'data/interim/wikidata5m_human_facts_that_have_labels_31032022_v3.tsv'),
#                                       sep = '\t', header = False, index = False)

# %% Step 10: Apply changes after removing entities/relations to the other files

# 1,743,638 QIDs remain
remaining_tails = set(human_triples_that_have_labels['tail_entity'])
remaining_heads = set(human_triples_that_have_labels['head_entity'])
remaining_QIDs = remaining_tails.union(remaining_heads)
# 294 PIDs remain
remaining_PIDs = set(human_triples_that_have_labels['relation'])

### Remove the QIDs with no truthy label from the human entities file

old_human_entities_file = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5m_human_entities_040122_v1.tsv'),
    header = None).squeeze('columns')

# This removes 6272 entities --> the majority of the overall 6566 QIDs!
human_entities_to_remove_mask = old_human_entities_file.isin(remaining_QIDs)
new_human_entities_file = old_human_entities_file[human_entities_to_remove_mask]

# save the new version of the file
new_human_entities_file.to_csv(
    os.path.join(BASE_PATH_HOST, 'data/interim/wikidata5m_human_entities_310322_v2.tsv'),
    sep = '\t', header = False, index = False)

### Filter the entity2text and relation2text files created in step 8
# Python objects: all_entity_labels, all_relation_labels
entity_labels_for_my_W5M_subset_mask = all_entity_labels['ID'].isin(remaining_QIDs)
entity_labels_for_my_W5M_subset = all_entity_labels[entity_labels_for_my_W5M_subset_mask]

# I expect that I get a label from the large file for each QID in my subset.
assert len(entity_labels_for_my_W5M_subset) == len(remaining_QIDs)
number_of_entity_labels_that_are_not_needed = len(all_entity_labels) - len(entity_labels_for_my_W5M_subset)
# 80041813 QID labels can be discarded.

# do the same for the relation labels
relation_labels_for_my_W5M_subset_mask = all_relation_labels['ID'].isin(remaining_PIDs)
relation_labels_for_my_W5M_subset = all_relation_labels[relation_labels_for_my_W5M_subset_mask]

# I expect that I get a label from the large file for each QID in my subset.
assert len(relation_labels_for_my_W5M_subset) == len(remaining_PIDs)
number_of_relation_labels_that_are_not_needed = len(all_relation_labels) - len(relation_labels_for_my_W5M_subset)
# 9207 PID labels can be discarded.

# Save new entity2text and relation2text files
# entity_labels_for_my_W5M_subset.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data/interim/truthy_triples_file_for_W5M_gender_facts/entity2label_W5M_truthy_31032022_v2.tsv'),
#     sep = '\t', header = False, index = False)
# relation_labels_for_my_W5M_subset.to_csv(
#     os.path.join(BASE_PATH_HOST, 'data/interim/truthy_triples_file_for_W5M_gender_facts/relation2label_W5M_truthy_31032022_v2.tsv'),
#     sep = '\t', header = False, index = False)

# %% Step 11: Create the files needed for KG-BERT

folder_to_save = os.path.join(BASE_PATH_HOST, 'data/interim/KG_and_LM_files')

### entities.txt: list of all entities in the dataset
with open(os.path.join(folder_to_save, 'entities_31032022_v1.txt'), 'w') as f:
    f.writelines(''.join([str(x) + '\n' for x in remaining_QIDs]))
f.close()

### relations.txt: list of all relations in the dataset
with open(os.path.join(folder_to_save, 'relations_31032022_v1.txt'), 'w') as f:
    f.writelines(''.join([str(x) + '\n' for x in remaining_PIDs]))
f.close()

### entity2text.tsv: 1 column QIDs, one column label, i.e. short text version
# simple copy and rename the file created in the previous step
# file name: entity2label_31032022_v1.tsv

### relation2text.tsv: 1 column PIDs, one column label, i.e. short text version
# simple copy and rename the file created in the previous step
# file name: relation2label_31032022_v1.tsv

### entity2textlong.tsv: filter down the original Wikidata5M description file
entity_descriptions = pd.read_csv(os.path.join(BASE_PATH_HOST, 'data/raw/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_wikipedia_texts.txt'),
                                  sep = '\t', names = ['ID', 'description'])
# doublecheck the import
entity_descriptions.head()
entity_descriptions.info()
len(entity_descriptions['ID'].unique()) == len(entity_descriptions)

# only keep the QID descriptions that are in my human facts subset
my_subset_of_entity_descriptions_mask = entity_descriptions['ID'].isin(remaining_QIDs)
my_subset_of_entity_descriptions = entity_descriptions[my_subset_of_entity_descriptions_mask]

# I expect that I get a description from the large file for each QID in my subset.
assert len(my_subset_of_entity_descriptions) == len(remaining_QIDs)

number_of_entity_descriptions_that_are_missing = len(remaining_QIDs) - len(my_subset_of_entity_descriptions)

# check whether the missing QIds were part of the original Wikidata5M human subset



### TODO train.tsv, dev.tsv, test.tsv --> decide for a split

# Keidar: did an 0.8 0.1 0.1 split
# with about 9 million human facts, this results in 900,000 valid + test triples

# KEPLER paper: in transductive split, the validation + test set is "tiny"
# valid: only 5163 triples of 20624575 triples (0.025%)
# test: only 5133 triples of 20624575 triples (0.02489%)


# %% TODO Create utility file for KG only: map from numeric ID to string ID to text label

# first column: numeric ID starting at 0
# second column: entity/relation ID
# third column: English label (extracted from truthy file)

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

# %% TODO handle the Unicode literals in the label file

# %% TODO Decide how to treat the rare triples, e.g. count > 5

# data/interim/tail_value_counts_human_facts_W5M_8.2.2022.csv
# exploration/relationship_counts/tail_value_counts_all_11.11.2021.csv


# %% TODO alternative: keep existing splits and use only human facts

# check whether doing this removes triple uniformly from all 3 splits


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

# %% Exploration: explore the alias files from the original Wikidata5M files

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


# %% Exploration: check whether W5M in pykeen contains short labels

import pykeen
from pykeen.datasets import Wikidata5M

# there are no text labels, only numeric IDs!
# after all it is only considered a knowledge graph dataset here...
w5m = Wikidata5M()
w5m.relation_to_id
w5m.entity_to_id

# %% Exploration: check whether W5M in graphvite contains short labels
# conclusion: I can't install Graphvite properly, so I can't check it

# the documentation mentions the aliases, i.e. natural language labels
# source: https://graphvite.io/docs/latest/pretrained_model
# "Load the alias mapping from the dataset.
# Now we can access the embeddings by natural language index."

import graphvite as gv

alias2entity = gv.dataset.wikidata5m.alias2entity
alias2relation = gv.dataset.wikidata5m.alias2relation  # print(entity_embeddings[entity2id[alias2entity["machine learning"]]])
# print(relation_embeddings[relation2id[alias2relation["field of work"]]])


# %% Exploration: Do the relation labels if Wikidata5M and truthy triples largely agree?

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
