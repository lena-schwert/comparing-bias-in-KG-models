# import from built-in modules
import os
import sys
import time
import gc

import pandas as pd
import numpy as np
import seaborn as sns

from SPARQLWrapper import SPARQLWrapper, JSON

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 100)

# %% import spreadsheet sensitive relation encoding as pandas dataframe

# file from 22.10. includes Wikidata P-IDs and their encoding by Wikidatasets-human
# file from 25.10. additionally includes OpenKE IDs
# or use target relations dataset
sensitive_properties = pd.read_csv(
    '/home/lena/git/master_thesis_bias_in_NLP/exploration/relationship_counts/sensitive_Wikidata_relations_25.10.2021.csv',
    na_values = "NA",
    dtype = {'P_ID': str, 'wikidata_label': str, 'sensitive_attribute': 'category'})
# this is necessary to cast "Wikidatasets_ID" to Int64 and labels to pandas string type
sensitive_properties = sensitive_properties.convert_dtypes()

target_properties = pd.read_csv(
    '/home/lena/git/master_thesis_bias_in_NLP/exploration/relationship_counts/target_Wikidata_relations_25.10.2021.csv',
    na_values = "NA",
    dtype = {'P_ID': str, 'wikidata_label': str, 'sensitive_attribute': 'category'})
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

# %% Utility functions for making relation counts


def get_triples_df(name_of_dataset_processed):
    # do specific things for different datasets, if required
    if name_of_dataset_processed.lower() == 'wikidata5m':
        # file names: e.g. wikidata5m_all_triplets.txt
        # each line is a triple: Q29387131	P31	Q5 (tab-separated)
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/SOTA_datasets_raw_downloads/Wikidata5M/'
        file_name = 'wikidata5m_all_triplets.txt'
        triples_df = pd.read_csv(os.path.join(dataset_folder, file_name), sep = '\t',
                                 names = ['head_entity', 'relation', 'tail_entity'])
        property_encoding_ID = 'P_ID'

    elif name_of_dataset_processed.lower() == 'wikidatasets-humans':
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/Wikidatasets_humans/'
        # 44 million rows, needs 1GB RAM, rows are integers only
        # contains all triples where the tail entity is no human
        attributes_df = pd.read_csv(os.path.join(dataset_folder, 'attributes.tsv'), sep = '\t',
                                    names = ['head_entity', 'tail_entity', 'relation'],
                                    skiprows = 1)
        # 3.3 million rows, needs 75MB RAM, rows are integers only
        # contains all triples head and tail entity are human
        edges_df = pd.read_csv(os.path.join(dataset_folder, 'edges.tsv'), sep = '\t',
                               names = ['head_entity', 'tail_entity', 'relation'], skiprows = 1)

        # check whether there is any overlap between the two dataframes (there shouldn't be)
        # with merge, check whether first dataframe (edges) is in the second (attributes)
        pd.merge(edges_df.reset_index(), attributes_df, how = 'inner').set_index('index')
        # this returns an empty dataframe either way!

        # add both dataframes together
        triples_df = pd.concat([edges_df, attributes_df])

        property_encoding_ID = 'Wikidatasets_ID'

    elif name_of_dataset_processed.lower() == "openke":
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/OpenKE-Wikidata/knowledge graphs/'
        # each line is a triple: 0 1 0 (tab-separated)
        # 69 million rows, needs 1.5GB RAM, rows are integers only (same as Wikidatasets)
        # column ordering mentioned on Github
        triples_df = pd.read_csv(os.path.join(dataset_folder, 'triple2id.txt'), sep = '\t',
                                 names = ['head_entity', 'tail_entity', 'relation'], skiprows = 1)

        property_encoding_ID = 'OpenKE_ID'

    elif name_of_dataset_processed.lower() == 'codex-l' or name_of_dataset_processed.lower() == 'codex-m' or name_of_dataset_processed.lower() == 'codex-s':
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/Codex_S_M_L/triples/'
        train_triples = pd.read_csv(
            os.path.join(dataset_folder, name_of_dataset_processed.lower(), 'train.txt'),
            sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])
        valid_triples = pd.read_csv(
            os.path.join(dataset_folder, name_of_dataset_processed.lower(), 'valid.txt'),
            sep = '\t', names = ['head_entity', 'relation', 'tail_entity'])
        test_triples = pd.read_csv(
            os.path.join(dataset_folder, name_of_dataset_processed.lower(), 'test.txt'), sep = '\t',
            names = ['head_entity', 'relation', 'tail_entity'])
        # concatenate all individual data frame into a single one
        triples_df = pd.concat([train_triples, valid_triples, test_triples])
        # akternatively load raw triples  # triples_df = pd.read_csv(
        #     '/home/lena/git/master_thesis_bias_in_NLP/data/Codex_S_M_L/triples/raw_triples.txt',
        #      sep = '\t',
        #      names = ['head_entity', 'relation', 'tail_entity'])
        property_encoding_ID = 'P_ID'
    else:
        print(f"Dataset name {name_of_dataset_processed} not found!")

    return triples_df, property_encoding_ID


def load_dataset_return_relations_count(property_encoding_df, name_of_dataset_processed):
    print(f'Counting relations for {name_of_dataset_processed} dataset...')

    name_of_dataset_processed = name_of_dataset_processed.lower()

    # do specific things for different datasets, if required
    if name_of_dataset_processed == 'wikidata5m':
        triples_df, lookup_column_for_filtering = get_triples_df(name_of_dataset_processed)

        # How many times does "... instance of human" appear?
        number_of_human_entities = triples_df[(triples_df['relation'] == 'P31') & (triples_df[
                                                                                       'tail_entity'] == 'Q5')].__len__()  # 1519261 rows  # How many times does "... subclass of human" appear?  # can be neglected!  # triples_df[(triples_df['relation'] == 'P279') & (triples_df['tail_entity'] == 'Q5')]  # 84 rows


    elif name_of_dataset_processed == 'wikidatasets-humans':
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/Wikidatasets_humans/'
        triples_df, lookup_column_for_filtering = get_triples_df(name_of_dataset_processed)

        # How many human entities?
        nodes_df = pd.read_csv(os.path.join(dataset_folder, 'nodes.tsv'), sep = '\t',
                               names = ['entity_ID', 'wikidata_ID', 'label'], skiprows = 1)
        number_of_human_entities = nodes_df.index.__len__() + 1

    elif name_of_dataset_processed == 'openke':
        triples_df, lookup_column_for_filtering = get_triples_df(name_of_dataset_processed)

        # How many human entities?
        # instance of: 4, subclass of: 90, human, Q5: 68
        number_of_human_entities = triples_df[
            (triples_df['relation'] == 4) & (triples_df['tail_entity'] == 68)].__len__()

    elif name_of_dataset_processed == 'codex-l' or name_of_dataset_processed == 'codex-m' or name_of_dataset_processed == 'codex-s':
        triples_df, lookup_column_for_filtering = get_triples_df(name_of_dataset_processed)

        # How many times does "... instance of human" appear?
        number_of_human_entities = triples_df[
            (triples_df['relation'] == 'P31') & (triples_df['tail_entity'] == 'Q5')].__len__()

    else:
        ImportError(f"Dataset name {name_of_dataset_processed} not found!")

    # count occurrence of all relations as pandas series where relations = index type
    all_relation_counts = triples_df['relation'].value_counts()

    new_column_name = 'count_' + name_of_dataset_processed
    property_encoding_df[new_column_name] = ''

    # loop through P_ID column to find the relevant counts only
    # sensitive_properties_df = pandas Series with type int64 (and som NA values)
    for i, relation in enumerate(property_encoding_df[lookup_column_for_filtering]):
        label_relation = property_encoding_df['wikidata_label'][
            property_encoding_df[lookup_column_for_filtering] == relation]
        # print(f'Relation: {label_relation.iloc[0]}, ID: {relation}')

        # add a number to the count column whenever the relation actually exists in the dataset
        if relation in set(all_relation_counts.index):  # & pd.isna(relation) == False:
            # print('relation exists in the dataset!')
            property_encoding_df.loc[i, new_column_name] = all_relation_counts[relation]
            print(f'Count is: {all_relation_counts[relation]}')
        else:
            # we land here for NA values
            property_encoding_df.loc[
                i, new_column_name] = np.nan  # print('This relation does NOT exist in the dataset!')

    # count total number of triples
    total_number_of_triples = triples_df.index.__len__() + 1

    property_encoding_df = property_encoding_df.convert_dtypes()

    print(f'Count added to sensitive_properties_df!')
    return property_encoding_df, total_number_of_triples, number_of_human_entities


# %% Do everything for Wikidatasets-Humans

# relation_counts_Wikidatasets_Humans, total_number_of_triples_Wikidatasets, number_of_human_entities_Wikidatasets\
#     = load_dataset_return_relations_count(
#     property_encoding_df = sensitive_properties,
#     name_of_dataset_processed = 'Wikidatasets-Humans')
#
# #relation_counts_Wikidatasets_Humans.to_csv('exploration/tosave_Wikidatasets-Humans_target_counts.csv')
#
# %% Do everything for Wikidata5M

# relation_counts_Wikidata5M, total_number_of_triples_Wikidata5M, number_of_human_entities_Wikidata5M = load_dataset_return_relations_count(
#     property_encoding_df = sensitive_properties, name_of_dataset_processed = 'Wikidata5M')

# save as csv to disk
# relation_counts_Wikidata5M.to_csv('exploration/tosave_Wikidata5M_target_counts.csv')

# %% Do everything for OpenKE

# relation_counts_OpenKE, total_number_of_triples_OpenKE, number_of_human_entities_OpenKE = load_dataset_return_relations_count(
# property_encoding_df = sensitive_properties, name_of_dataset_processed = 'OpenKE')

# save as csv to disk
# relation_counts_OpenKE.to_csv('exploration/tosave_OpenKE_target_counts.csv')

# %% Do everything for the 3 Codex datasets

# relation_counts_codex, total_number_of_triples_codex, number_of_human_entities_codex = load_dataset_return_relations_count(
#     property_encoding_df = sensitive_properties, name_of_dataset_processed = 'CoDEx-L')


# %% Make tail value counts for all datasets for sensitive + target relations

# TODO account for different occurrence of tail values across datasets

# idea: join the all_properties dataframe together with a dataset that is created below
# final dataset columns: dataset name, relation, tail value, counts (NA or integer)

list_of_all_datasets = ['Wikidata5M', 'OpenKE', 'Wikidatasets-Humans', 'Codex-L', 'Codex-M',
                        'Codex-S']

# create dataframe for storing the results
results_tail_value_counts = pd.DataFrame(
    columns = ['dataset_name', 'relation_P_ID', 'relation_label', 'tail_entity_Q_ID',
               'tail_entity_label', 'count'])

# loop through datasets one by one
for dataset in list_of_all_datasets:
    print(f'##### Retrieving tail value counts for {dataset} dataset...')
    # retrieve the triples dataset + data-specific variable for filtering the relation column
    triples_df, property_encoding_ID = get_triples_df(dataset)

    # if necessary, load data-specific ID to Q-ID mapping
    if dataset == 'Wikidatasets-Humans':
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/Wikidatasets_humans/'
        Q_IDs_to_labels = pd.read_csv(os.path.join(dataset_folder, 'entities.tsv'),
                                      sep = '\t', skiprows = 1,
                                      names = ['dataset_id', 'wikidata_qid', 'wikidata_label'])
        Q_IDs_to_labels.drop('wikidata_label', axis = 1, inplace = True)

    if dataset == 'OpenKE':
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/OpenKE-Wikidata/knowledge graphs/'
        Q_IDs_to_labels = pd.read_csv(os.path.join(dataset_folder, 'entity2id.txt'), sep = '\t',
                                  names = ['wikidata_qid', 'dataset_id'], skiprows = 1)
        # make sure that first column is dataset-spcific ID
        new_column_titles = ['dataset_id', 'wikidata_qid']
        Q_IDs_to_labels = Q_IDs_to_labels.reindex(columns = new_column_titles)

    # create a subset of the dataset using the relations in all properties
    list_of_all_relations = all_properties_df[property_encoding_ID]

    # list_of_all_relations_regex_OR = '|'.join(list_of_all_relations)
    # # create filtering mask and preserve original indices
    # filtering_mask_all_relations_match = triples_df.stack().str.match(list_of_all_relations_regex_OR).groupby(level = 0).any()
    # triples_df_only_selected_relations_match = triples_df[filtering_mask_all_relations]
    # # has 5116730 entries
    # filtering_mask_all_relations_fullmatch = triples_df.stack().str.fullmatch(list_of_all_relations_regex_OR).groupby(level = 0).any()
    # triples_df_only_selected_relations_fullmatch = triples_df[filtering_mask_all_relations_fullmatch]
    # # has 4749166 entries

    # alternative: use isin()
    filtering_mask_all_selected_relations = triples_df['relation'].isin(list_of_all_relations)
    triples_df_all_selected_relations = triples_df[filtering_mask_all_selected_relations]
    # has 4749166 entries
    # store relations of the dataset as a set
    all_relations_present_in_dataset = set(triples_df_all_selected_relations['relation'].unique())

    # free up memory
    del triples_df
    gc.collect()

    # retrieve tail counts for each of the relations found in the dataset
    for i, relation in enumerate(all_properties_df[property_encoding_ID]):
        print(f'Counting tail values for relation: {relation}')

        # account for relations that do not occur in this particular dataset
        if relation in all_relations_present_in_dataset:
            # for each relation, make a subset of the subset dataset
            filtering_mask_only_current_relation = triples_df_all_selected_relations[
                'relation'].isin([relation])
            triples_df_only_current_relation = triples_df_all_selected_relations[
                filtering_mask_only_current_relation]

            # make value counts across the tail entities for this dataframe
            triples_df_only_current_relation_value_counts = triples_df_only_current_relation[
                'tail_entity'].value_counts()

            relation_P_ID = relation
            relation_Q_IDs = triples_df_only_current_relation_value_counts.index.values  # np.array
            # if necessary, transform between dataset-specific IDs and P/Q-IDs
            if property_encoding_ID != 'P_ID':
                # this is necessary for Wikidatasets-Humans and OpenKE
                # retrieve P_ID for dataset-specific ID
                relation_P_ID = all_properties_df['P_ID'][
                    all_properties_df[property_encoding_ID] == relation].item()
                # retrieve Q-IDs for each of the tail entities using Q_IDs_to_labels
                # use Index from value counts directly to access Q_IDs
                relation_Q_IDs = Q_IDs_to_labels['wikidata_qid'][triples_df_only_current_relation_value_counts.index].values

            # add this dataframe to the final results dataframe with dataset + relation name
            number_of_new_rows = len(triples_df_only_current_relation_value_counts)
            rows_to_add = pd.DataFrame({'dataset_name': [dataset] * number_of_new_rows,
                                        'relation_P_ID': [relation_P_ID] * number_of_new_rows,
                                        'relation_label': [all_properties_df['wikidata_label'][
                                                               all_properties_df[
                                                                   'P_ID'] == relation_P_ID].item()] * number_of_new_rows,
                                        'tail_entity_Q_ID': relation_Q_IDs,
                                        'tail_entity_label': ['NA'] * number_of_new_rows,
                                        'count': triples_df_only_current_relation_value_counts.values})
            results_tail_value_counts = pd.concat([results_tail_value_counts, rows_to_add])
        else:
            # Which relations I selected do not occur in this dataset?
            # relations_not_contained = set(all_properties_df['P_ID']) - all_relations_present_in_dataset
            print('This relation is not contained in the current dataset!')
            # These are:
            # Wikidata5M: place of birth
            # OpenKE: social classification
            # account for non-numeric relations in NA special case
            if isinstance(relation, str) == False:
                relation = all_properties_df['P_ID'].loc[i]

            # add a NA row to the results dataframe
            NA_row_to_add = pd.DataFrame(
                {'dataset_name': [dataset], 'relation_P_ID': [relation],
                 'relation_label': [all_properties_df['wikidata_label'][
                                                               all_properties_df[
                                                                   'P_ID'] == relation].item()],
                 'tail_entity_Q_ID': ['NA'], 'tail_entity_label': ['NA'], 'count': [np.nan]})
            results_tail_value_counts = pd.concat([results_tail_value_counts, NA_row_to_add])

    # free up RAM for next dataset
    del triples_df_all_selected_relations
    gc.collect()

# reindex entire results dataframe
results_tail_value_counts.reset_index(drop = True, inplace = True)

results_tail_value_counts = results_tail_value_counts.convert_dtypes()

print('Collected all counts!')
print('yay')

# write dataframe to csv/pickle
results_tail_value_counts.to_csv('tail_value_counts_all_11.11.2021.csv')
results_tail_value_counts.to_pickle('tail_value_counts_all_11.11.2021.pkl')


# %% Extract sensitive relation counts from current Wikidata via SPARQL

endpoint_url = "https://query.wikidata.org/sparql"


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent = user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


# assign new column to dataset
sparql_new_column_name = 'sparql_count_14.10.2021'
sensitive_properties[sparql_new_column_name] = ''
sensitive_properties['sparql_query_time_s'] = ''

# loop through P_ID column
for i, sensitive_p in enumerate(sensitive_properties['P_ID']):
    print(f'Relation queried: {sensitive_p}')
    # print(type(sensitive_p))
    # add the current P-ID in the parametrized f-string of the query
    # query_female_human_count = f'SELECT (COUNT (DISTINCT ?item) AS ?count) WHERE {{?item wdt:{sensitive_p} wd:Q6581072. SERVICE wikibase:label {{bd: serviceParam wikibase: language "en".}} }}'
    SPARQL_query = f'SELECT (COUNT (DISTINCT ?item) AS ?count) WHERE {{?item wdt:P31 wd:Q5; wdt:{sensitive_p} ?value. }}'

    print('Query has started.')
    start_time = time.perf_counter()
    try:
        results = get_results(endpoint_url, SPARQL_query)

    except:
        results = None
        print("Server did not return results, moving on to next relation.")
    end_time = time.perf_counter()
    query_time_s = round(end_time - start_time, 2)
    print(f'This query took {query_time_s} seconds.')
    if results:
        # transform to pandas dataframe
        results_df = pd.json_normalize(results.get('results').get('bindings'))
        # add to dataframe
        sensitive_properties.loc[i, sparql_new_column_name] = int(results_df['count.value'])
        sensitive_properties.loc[i, 'sparql_query_time_s'] = query_time_s
    else:
        sensitive_properties.loc[i, sparql_new_column_name] = np.nan
        sensitive_properties.loc[i, 'sparql_query_time_s'] = np.nan

sensitive_properties = sensitive_properties.convert_dtypes()
sensitive_properties.info()

# write file to disk
print('Dataframe from SPARQL was created.')
# os.getcwd()
sensitive_properties.to_csv('tosave_SPARQL_counts_RENAME_ME.csv')
