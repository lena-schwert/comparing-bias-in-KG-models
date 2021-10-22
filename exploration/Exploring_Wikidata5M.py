import os
import sys
import time


import pandas as pd
import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON, CSV

# %% import spreadsheet as pandas dataframe
sensitive_properties = pd.read_csv(
    '/home/lena/git/master_thesis_bias_in_NLP/exploration/sensitive_Wikidata_relations_14.10.2021.tsv',
    sep = '\t', dtype = {'P_ID': str, 'wikidata_label': str, 'sensitive_attribute': 'category'})

# %% Load the triple file into pandas

# What kind of file is it?
# - txt file
# - 2,3 GB large
# - each line is a triple: Q29387131	P31	Q5
# helpful characters: tab-separated

# read as TSV
file_path = '/home/lena/git/master_thesis_bias_in_NLP/data/SOTA datasets_raw downloads/Wikidata5M/'
dataset_name = 'wikidata5m_transductive_test'
file_ending = '.txt'
test = pd.read_csv(os.path.join(file_path, dataset_name) + file_ending, sep = '\t',
                   names = ['head_entity', 'relation', 'tail_entity'])

# test: count occurrences of relation
relation_counts = test[
    'relation'].value_counts()  # is a pandas series, relations = index type (not so practical)

# %% Loop through all sensitive attribute

sensitive_properties.info()

new_column_name = 'count_' + dataset_name
sensitive_properties[new_column_name] = ''

# loop through P_ID column
for i, sensitive_p in enumerate(sensitive_properties['P_ID']):
    print(sensitive_p)
    # do a cheap check whether the ID actually exists (set?)
    if sensitive_p in set(relation_counts.index):
        print('relation exists in the dataset!')
        sensitive_properties.loc[i, new_column_name] = relation_counts[sensitive_p]
    else:
        sensitive_properties.loc[i, new_column_name] = np.nan
        print('relation does NOT exist in the dataset!')

sensitive_properties = sensitive_properties.convert_dtypes()
sensitive_properties.info()
print('dataset created')

# %% write file to disk

sensitive_properties.to_csv('tosave_train,valid,test,all_counts.csv')

# %% Extract sensitive relation counts from current Wikidata via SPARQL

endpoint_url = "https://query.wikidata.org/sparql"

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent = user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

# %% parametrize the query using the list of sensitive relations

# assign new column to dataset
sparql_new_column_name = 'sparql_count_14.10.2021'
sensitive_properties[sparql_new_column_name] = ''
sensitive_properties['sparql_query_time_s'] = ''

# loop through P_ID column
for i, sensitive_p in enumerate(sensitive_properties['P_ID']):
    print(sensitive_p)
    print(type(sensitive_p))
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
print('Dataframe from SPARQL was created.')

# %% write file to disk

os.getcwd()
sensitive_properties.to_csv('tosave_SPARQL_counts.csv')