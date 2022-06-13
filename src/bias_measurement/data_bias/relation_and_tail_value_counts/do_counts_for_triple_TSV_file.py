# import from built-in modules
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


# %% import spreadsheet sensitive relation encoding as pandas dataframe

# file from 22.10. includes Wikidata P-IDs and their encoding by Wikidatasets-human
# file from 25.10. additionally includes OpenKE IDs
# or use target relations dataset
sensitive_properties = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                'exploration/relationship_counts/sensitive_Wikidata_relations_25.10.2021.csv'),
                                   na_values = "NA", dtype = {'P_ID': str, 'wikidata_label': str,
                                                              'sensitive_attribute': 'category'})
# this is necessary to cast "Wikidatasets_ID" to Int64 and labels to pandas string type
sensitive_properties = sensitive_properties.convert_dtypes()

target_properties = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                             'exploration/relationship_counts/target_Wikidata_relations_25.10.2021.csv'),
                                na_values = "NA", dtype = {'P_ID': str, 'wikidata_label': str,
                                                           'sensitive_attribute': 'category'})
# this is necessary to cast "Wikidatasets_ID" to Int64 and labels to pandas string type
target_properties = target_properties.convert_dtypes()

all_properties_df = pd.concat([sensitive_properties, target_properties])
all_properties_df.reset_index(inplace = True, drop = True)

# drop some relations from the dataframe, because they have too low counts
# drop 9 relations overall
relations_to_drop = ['date of birth', 'date of death', 'personal pronoun', 'permanent resident of',
                     'facial hair', 'hair style', 'religion or world view', 'diaspora',
                     'nominated by']

# drop these rows from the all_properties_df
filter = set(relations_to_drop)
to_delete = list()

for id, row in all_properties_df.iterrows():
    current_item = set([row.wikidata_label])
    if current_item.intersection(filter):
        to_delete.append(id)

all_properties_df.drop(to_delete, inplace = True)
all_properties_df.reset_index(inplace = True, drop = True)

# IMPORTANT: drop the WIkidatasets ID and OpenKE ID
all_properties_df.drop(['Wikidatasets_ID', 'OpenKE_ID'], axis = 1, inplace = True)

# %% Decide what should be done + set paths

START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")

DO_RELATION_COUNTS = False

DO_TAIL_VALUE_COUNTS = True

dataset_name = 'FB15K-237'

if dataset_name == 'HumanWikidata5M':
    path_to_directory = 'data/processed/output_of_preprocessing'
    file_name = 'test_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
    #test_data_subset_0.05_rs42_06_05_2022_15:11.tsv
    #wikidata5M_human_facts_subset_060522_v4.tsv
    #wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv
    property_encoding_ID = 'P_ID'

if dataset_name == 'FB15K-237':
    #path_to_directory = 'data/processed/files_per_model/KG_and_LM_KGBERT_FB15K-237/'
    path_to_directory = 'data/processed/output_of_preprocessing'
    file_name = 'FB15K237_complete.tsv'
    #file_name = 'FB15K237_testset_only_gender_occupation_facts_complete.tsv'
    property_encoding_ID = 'Freebase_ID'

# %% Utility functions for making relation counts


def get_triples_df(relative_path_to_TSV_file):
    # do specific things for different datasets, if required

    # each line is a triple: Q29387131	P31	Q5 (tab-separated)
    path_to_file = os.path.join(BASE_PATH_HOST, relative_path_to_TSV_file)
    triples_df = pd.read_csv(path_to_file,
                             sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])

    return triples_df


def load_dataset_return_relations_count(property_encoding_df: pd.DataFrame, relative_path_to_TSV_file: str,
                                        property_encoding_ID: str, dataset_name: str):
    print(f'Counting relations for file located at:  {relative_path_to_TSV_file}')

    triples_df = get_triples_df(relative_path_to_TSV_file)

    if dataset_name == 'HumanWikidata5M':
        # How many times does "... instance of human" appear?
        number_of_human_entities = triples_df[(triples_df['relation'] == 'P31') & (triples_df[
                                                        'tail_entity'] == 'Q5')].__len__()
    if dataset_name == 'FB15K-237':
        number_of_human_entities = None

    # count occurrence of all relations as pandas series where relations = index type
    all_relation_counts = triples_df['relation'].value_counts()

    new_column_name = 'count'
    property_encoding_df[new_column_name] = ''

    # loop through P_ID column to find the relevant counts only
    # sensitive_properties_df = pandas Series with type int64 (and som NA values)
    for i, relation in enumerate(property_encoding_df[property_encoding_ID]):
        label_relation = property_encoding_df['wikidata_label'][
            property_encoding_df[property_encoding_ID] == relation]

        # add a number to the count column whenever the relation actually exists in the dataset
        if relation in set(all_relation_counts.index):  # & pd.isna(relation) == False:
            # print('relation exists in the dataset!')
            property_encoding_df.loc[i, new_column_name] = all_relation_counts[relation]
            print(f'Count for relation {label_relation} is: {all_relation_counts[relation]}')
        else:
            # we land here for NA values
            property_encoding_df.loc[
                i, new_column_name] = np.nan  # print('This relation does NOT exist in the dataset!')

    # count total number of triples
    total_number_of_triples = triples_df.index.__len__() + 1

    property_encoding_df = property_encoding_df.convert_dtypes()

    property_encoding_df = property_encoding_df.sort_values('count', ascending = False)

    print(f'Count added to property_encoding_df!')
    return property_encoding_df, total_number_of_triples, number_of_human_entities


# %% Do the relation counts or all sensitive + target relations

if DO_RELATION_COUNTS:

    relation_counts, total_number_of_triples, number_of_human_entities = load_dataset_return_relations_count(
        property_encoding_df = all_properties_df,
        relative_path_to_TSV_file = os.path.join(path_to_directory, file_name),
        property_encoding_ID = property_encoding_ID,
        dataset_name = dataset_name
    )

    # save as csv to disk
    base_results_folder = 'results/bias_measurement/data_bias/relation_counts/'
    relation_counts.to_csv(os.path.join(BASE_PATH_HOST, base_results_folder, f'FB15K237_complete_relation_counts_{START_TIME}.tsv'),
                           sep = '\t', index = False)


# %% Do the tail value counts for all sensitive + target relations


# idea: join the all_properties dataframe together with a dataset that is created below
# final dataset columns: dataset name, relation, tail value, counts (NA or integer)

if DO_TAIL_VALUE_COUNTS:

    # IMPORTANT: If desired, only count tails for specific relations

    if dataset_name == 'FB15K-237':
        relations_to_extract = all_properties_df[
            all_properties_df['Freebase_ID'].isin(['/people/person/profession'])]
        IDs_to_labels = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                   'data/processed/files_per_model/KG_and_LM_KGBERT_FB15K-237/entity2text.txt'),
                                    sep = '\t', names = ['ID', 'label'])

    if dataset_name == 'HumanWikidata5M':
        relations_to_extract = all_properties_df[all_properties_df['P_ID'].isin(['P106'])]
        # relations_to_extract = all_properties_df[all_properties_df['P_ID'].isin(['P69'])]
        # relations_to_extract = all_properties_df[all_properties_df['P_ID'].isin(['P169'])]

        # use the entity2label file created from the truthy triples file
        IDs_to_labels = pd.read_csv(os.path.join(BASE_PATH_HOST, 'data/processed/output_of_preprocessing',
                                                   'entity2label_wikidata5m_human_clean_utf8_11052022_v3.tsv'),
                                    sep = '\t', names = ['ID', 'label'])

    # create dataframe for storing the results
    results_tail_value_counts = pd.DataFrame(
        columns = ['relation_ID', 'relation_label', 'tail_entity_ID',
                   'tail_entity_label', 'count'])

    print(f'##### Retrieving tail value counts for file located at: {os.path.join(path_to_directory, file_name)}')
    # retrieve the triples dataset + data-specific variable for filtering the relation column
    triples_df = get_triples_df(os.path.join(path_to_directory, file_name))

    # create a subset of the dataset using the relations in all properties
    list_of_all_relation_IDs = relations_to_extract[property_encoding_ID]

    # create subset of triples_df using list_of_all_relation_IDs
    filtering_mask_all_selected_relations = triples_df['relation'].isin(list_of_all_relation_IDs)
    triples_df_all_selected_relations = triples_df[filtering_mask_all_selected_relations]
    print(f'{len(triples_df_all_selected_relations)} triples were selected using list_of_all_relation_IDs.')

    # store relations of the dataset as a set
    set_of_relations_in_triples_df_all_selected_relations = set(triples_df_all_selected_relations['relation'].unique())

    # free up memory
    del triples_df
    gc.collect()

    # retrieve tail counts for each of the relations found in the dataset
    for i, relation in enumerate(relations_to_extract[property_encoding_ID]):
        print(f'Counting tail values for relation: {relation}')

        # account for relations that do not occur in this particular dataset
        if relation in set_of_relations_in_triples_df_all_selected_relations:
            # for each relation, make a subset of the subset dataset
            filtering_mask_only_current_relation = triples_df_all_selected_relations[
                'relation'].isin([relation])
            triples_df_only_current_relation = triples_df_all_selected_relations[
                filtering_mask_only_current_relation]

            # make value counts across the tail entities for this dataframe
            triples_df_only_current_relation_value_counts = triples_df_only_current_relation[
                'tail_entity'].value_counts()

            relation_P_ID = relation
            tail_entity_Q_IDs = triples_df_only_current_relation_value_counts.index.values  # np.array
            tail_entity_labels = [IDs_to_labels['label'][IDs_to_labels['ID'] == tail_entity].item() for tail_entity in tail_entity_Q_IDs]

            # add this dataframe to the final results dataframe with dataset + relation name
            number_of_new_rows = len(triples_df_only_current_relation_value_counts)
            assert len(tail_entity_Q_IDs) == number_of_new_rows
            assert len(tail_entity_labels) == number_of_new_rows



            rows_to_add = pd.DataFrame({'file_name': [file_name] * number_of_new_rows,
                                        'relation_ID': [relation_P_ID] * number_of_new_rows,
                                        'relation_label': [relations_to_extract['wikidata_label'][
                                                               relations_to_extract[
                                                                   property_encoding_ID] == relation_P_ID].item()] * number_of_new_rows,
                                        'tail_entity_ID': tail_entity_Q_IDs,
                                        'tail_entity_label': tail_entity_labels,
                                        'count': triples_df_only_current_relation_value_counts.values})
            results_tail_value_counts = pd.concat([results_tail_value_counts, rows_to_add])

        else:
            # Which relations I selected do not occur in this dataset?
            # relations_not_contained = set(relations_to_extract['P_ID']) - set_of_relations_in_triples_df_all_selected_relations
            print('This relation is not contained in the current dataset!')
            # These are:
            # Wikidata5M: place of birth
            # OpenKE: social classification
            # account for non-numeric relations in NA special case
            if isinstance(relation, str) == False:
                relation = relations_to_extract['P_ID'].loc[i]

            # add a NA row to the results dataframe
            NA_row_to_add = pd.DataFrame({'file_name': [file_name], 'relation_P_ID': [relation],
                                          'relation_label': [relations_to_extract['wikidata_label'][
                                                                 relations_to_extract[
                                                                     'P_ID'] == relation].item()],
                                          'tail_entity_Q_ID': ['NA'], 'tail_entity_label': ['NA'],
                                          'count': [np.nan]})
            results_tail_value_counts = pd.concat([results_tail_value_counts, NA_row_to_add])

    # free up RAM for next dataset
    del triples_df_all_selected_relations
    gc.collect()

    # reindex entire results dataframe
    results_tail_value_counts.reset_index(drop = True, inplace = True)

    results_tail_value_counts = results_tail_value_counts.convert_dtypes()

    print('Collected all tail value counts!')
    print('yay')

    # filter for count threshold
    #counts_geq_5 = results_tail_value_counts.query('count>=5')

    # write dataframe to csv/pickle
    base_results_folder = 'results/bias_measurement/data_bias/tail_value_counts/'
    results_tail_value_counts.to_csv(os.path.join(BASE_PATH_HOST, base_results_folder,
                                     f'tail_value_counts_profession_{file_name}_{START_TIME}.tsv'), index = False,
                                     sep = '\t')


