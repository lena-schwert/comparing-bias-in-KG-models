import os
import sys
import time

import pandas as pd
import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON, CSV

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# %% import spreadsheet sensitive relation encoding as pandas dataframe

# file from 22.10. includes Wikidata P-IDs and their encoding by Wikidatasets-human
# file from 25.10. additionally includes OpenKE IDs
# or use target relations dataset
sensitive_properties = pd.read_csv(
    '/home/lena/git/master_thesis_bias_in_NLP/exploration/target_Wikidata_relations_25.10.2021.csv',
    na_values = "NA",
    dtype = {'P_ID': str, 'wikidata_label': str, 'sensitive_attribute': 'category'})

# this is necessary to cast "Wikidatasets_ID" to Int64
sensitive_properties = sensitive_properties.convert_dtypes()


# %% import target relations spreadsheet

# TODO count target relations for all datasets


# %% Utility function for making relation counts


def load_dataset_return_relations_count(property_encoding_df, name_of_dataset_processed):
    print(f'Counting relations for {name_of_dataset_processed} dataset...')

    # do specific things for different datasets, if required
    if name_of_dataset_processed == 'Wikidata5M':
        # file names: e.g. wikidata5m_all_triplets.txt
        # each line is a triple: Q29387131	P31	Q5 (tab-separated)
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/SOTA datasets_raw downloads/Wikidata5M/'
        file_name = 'wikidata5m_all_triplets.txt'
        triples_df = pd.read_csv(os.path.join(dataset_folder, file_name), sep = '\t',
                                 names = ['head_entity', 'relation', 'tail_entity'])
        # How many times does "... instance of human" appear?
        number_of_human_entities = triples_df[(triples_df['relation'] == 'P31') & (
                    triples_df['tail_entity'] == 'Q5')].__len__()  # 1519261 rows
        # How many times does "... subclass of human" appear?
        # can be neglected!
        # triples_df[(triples_df['relation'] == 'P279') & (triples_df['tail_entity'] == 'Q5')]  # 84 rows
        lookup_column_for_filtering = 'P_ID'

    elif name_of_dataset_processed == 'Wikidatasets-Humans':
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

        # How many human entities?
        nodes_df = pd.read_csv(os.path.join(dataset_folder, 'nodes.tsv'), sep = '\t',
                               names = ['entity_ID', 'wikidata_ID', 'label'], skiprows = 1)
        number_of_human_entities = nodes_df.index.__len__() + 1

        lookup_column_for_filtering = 'Wikidatasets_ID'

    elif name_of_dataset_processed == "OpenKE":
        dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/OpenKE-Wikidata/knowledge graphs/'
        # each line is a triple: 0 1 0 (tab-separated)
        # 69 million rows, needs 1.5GB RAM, rows are integers only (same as Wikidatasets)
        # column ordering mentioned on Github
        triples_df = pd.read_csv(os.path.join(dataset_folder, 'triple2id.txt'), sep = '\t',
                                 names = ['head_entity', 'tail_entity', 'relation'], skiprows = 1)

        # How many human entities?
        # instance of: 4, subclass of: 90, human, Q5: 68
        number_of_human_entities = triples_df[
            (triples_df['relation'] == 4) & (triples_df['tail_entity'] == 68)].__len__()

        lookup_column_for_filtering = 'OpenKE_ID'
        pass

    else:
        print(f"Dataset name {name_of_dataset_processed} not found!")

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

    # TODO count number of humans in the dataset

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
# # %% Do everything for Wikidata5M
#
# relation_counts_Wikidata5M, total_number_of_triples_Wikidata5M, number_of_human_entities_Wikidata5M = load_dataset_return_relations_count(
#     property_encoding_df = sensitive_properties, name_of_dataset_processed = 'Wikidata5M')
#
# # save as csv to disk
# #relation_counts_Wikidata5M.to_csv('exploration/tosave_Wikidata5M_target_counts.csv')

# %% Do everything for OpenKE

relation_counts_OpenKE, total_number_of_triples_OpenKE, number_of_human_entities_OpenKE = load_dataset_return_relations_count(
    property_encoding_df = sensitive_properties, name_of_dataset_processed = 'OpenKE')

# save as csv to disk
#relation_counts_OpenKE.to_csv('exploration/tosave_OpenKE_target_counts.csv')

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
    #print(type(sensitive_p))
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
#os.getcwd()
sensitive_properties.to_csv('tosave_SPARQL_counts_RENAME_ME.csv')
