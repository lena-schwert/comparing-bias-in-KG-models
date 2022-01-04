# Imports
import os
import socket

import pandas as pd


def set_base_path_based_on_host():
    """
    Using socket, this function
    Use this at the beginning of every script!
    :return: string that is the base path of the synced git folder
    """
    if socket.gethostname() == 'Schlepptop':
        base_dir = '/home/lena/git/master_thesis_bias_in_NLP/'
    # covers all CPU nodes
    elif 'node' in socket.gethostname():
        base_dir = '/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis'
    return base_dir


def get_triples_df(name_of_dataset_processed):
    """
    For each dataset, read in the

    Also returns property_encoding_ID, which is helpful

    :param name_of_dataset_processed: string that is the dataset name
    :return: tuple of dataframe with all triples
    """
    base_path = set_base_path_based_on_host()

    # do specific things for different datasets, if required
    if name_of_dataset_processed.lower() == 'wikidata5m':
        # file names: e.g. wikidata5m_all_triplets.txt
        # each line is a triple: Q29387131	P31	Q5 (tab-separated)
        dataset_folder = os.path.join(base_path, 'data/SOTA_datasets_raw_downloads/Wikidata5M/')
        file_name = 'wikidata5m_all_triplets.txt'
        triples_df = pd.read_csv(os.path.join(dataset_folder, file_name), sep = '\t',
                                 names = ['head_entity', 'relation', 'tail_entity'])
        property_encoding_ID = 'P_ID'

    elif name_of_dataset_processed.lower() == 'wikidatasets-humans':
        dataset_folder = os.path.join(base_path, 'data/Wikidatasets_humans/')
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
        dataset_folder = os.path.join(base_path, 'data/OpenKE-Wikidata/knowledge graphs/')
        # each line is a triple: 0 1 0 (tab-separated)
        # 69 million rows, needs 1.5GB RAM, rows are integers only (same as Wikidatasets)
        # column ordering mentioned on Github
        triples_df = pd.read_csv(os.path.join(dataset_folder, 'triple2id.txt'), sep = '\t',
                                 names = ['head_entity', 'tail_entity', 'relation'], skiprows = 1)

        property_encoding_ID = 'OpenKE_ID'

    elif name_of_dataset_processed.lower() == 'codex-l' or name_of_dataset_processed.lower() == 'codex-m' or name_of_dataset_processed.lower() == 'codex-s':
        dataset_folder = os.path.join(base_path, 'data/Codex_S_M_L/triples/')
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


def add_PID_labels_as_dfcolumn():
    # refer to Wikdiata_relation_counts.py from line 366
    raise NotImplementedError()


def add_PID_labels_as_dfcolumn():
    # refer to Wikdiata_relation_counts.py from line 366
    raise NotImplementedError()
