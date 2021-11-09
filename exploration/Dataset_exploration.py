import pandas as pd
import os

# %% Loop through relations file to find dataset-specific encodings for P IDs

sensitive_properties = pd.read_csv(
    '/home/lena/git/master_thesis_bias_in_NLP/exploration/target_Wikidata_relations_25.10.2021.csv',
    na_values = "NA",
    dtype = {'P_ID': str, 'wikidata_label': str})

# this is necessary to cast "Wikidatasets_ID" to Int64
sensitive_properties = sensitive_properties.convert_dtypes()

# for OpenKE datasets
dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/OpenKE-Wikidata/knowledge graphs/'
relationIDs_df = pd.read_csv(os.path.join(dataset_folder, 'relation2id.txt'), sep = '\t',
                             names = ['wikidata_ID', 'OpenKE_ID'])


# for Wikdatasets-Humans
dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/Wikidatasets_humans/'
relationIDs_df = pd.read_csv(os.path.join(dataset_folder, 'relations.tsv'), sep = '\t',
                             names = ['relation_ID', 'wikidata_ID', 'Wikidatasets-Humans_ID'], skiprows = 1)


columns = list(relationIDs_df.columns)
results = pd.DataFrame(columns = columns)
for i, sensitive_p in enumerate(sensitive_properties['P_ID']):
    #print(sensitive_p)
    code_for_sensitive_p = relationIDs_df[relationIDs_df['wikidata_ID'] == sensitive_p]
    results = results.append(code_for_sensitive_p, ignore_index = True)
    print(code_for_sensitive_p)

results.to_csv('intermediary_results_wikidatasets.csv')


# %% look at Wikidatasets-humans

dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/Wikidatasets_humans/'
# 44 million rows, needs 1GB RAM, rows are integers only
# contains all triples where the tail entity is no human
attributes_df = pd.read_csv(os.path.join(dataset_folder, 'attributes.tsv'), sep = '\t',
                            names = ['head_entity', 'tail_entity', 'relation'], skiprows = 1)
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

entities_df = pd.read_csv(os.path.join(dataset_folder, 'entities.tsv'), sep = '\t',
                          names = ['entity_ID', 'wikidata_ID', 'label'], skiprows = 1)

nodes_df = pd.read_csv(os.path.join(dataset_folder, 'nodes.tsv'), sep = '\t',
                       names = ['entity_ID', 'wikidata_ID', 'label'], skiprows = 1)

# %% Answering Questions about Wikidatasets-HUmans

# How many triples in total are contained?
triples_df.index.__len__() + 1  # 47306954

# How many overall entities in entities.csv?
entities_df.index.__len__() + 1  # 7949371

# How many human entities in nodes.csv?
nodes_df.index.__len__() + 1  # 6906693

# What is the label for human?
entities_df[entities_df['wikidata_ID'] == 'Q5']

# %% Answering Questions about OpenKE


# double-check the column order
# from relations2ID: maximal ID is 593
# max ID for each column?
triples_df.apply(max)  # column ordering is head, tail, relation (like Wikidatasets)

entities_df = pd.read_csv(os.path.join(dataset_folder, 'entity2id.txt'), sep = '\t',
                          names = ['Wikidata_ID', 'OpenKE_ID'], skiprows = 1)

entities_df[entities_df['Wikidata_ID'] == 'Q5']  # 48
