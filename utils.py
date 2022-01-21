# %% Imports
# IN-built functions
import logging
import os
import socket
import sys
from datetime import datetime

import numpy as np
# 3rd part modules
import pandas as pd
from pykeen.triples import TriplesFactory


# %% My custom helper functions across projects

def set_base_path_based_on_host():
    """
    Using socket, this function
    Use this at the beginning of every script!
    :return: string that is the base path of the synced git folder
    """
    if socket.gethostname() == 'Schlepptop':
        base_dir = '/home/lena/git/master_thesis_bias_in_NLP/'
    # covers all CPU + GPU nodes of the HPI
    elif 'node' in socket.gethostname() or socket.gethostname() in ['a6k5-01', 'dgxa100-01', 'ac922-01', 'ac922-02']:
        base_dir = '/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis'
    else:
        base_dir = None
        ValueError("Host name is not recognized!")
    assert type(base_dir) == str, "Path has not been set correctly as string, doublecheck!"
    return base_dir


def initialize_my_logger(file_name, file_mode: str = 'a'):
    """
    This function creates a customized logger based on Pythons built-in logging module.
    5 possible levels are: debug, info, warning, error, critical

    - print message to stdout *and* to file
    - logger level: DEBUG, logger will print messages of all levels!
    - each logger message starts with date + time
    - mode: append to existing file
    - uses the root Logger (don't know whether the alternative matters

    file_name: Name of file log.
    file_mode: Either 'w' to write new file [overwrite old log] or 'a' to append to recent log.


    Potential other features
    - add more information in the strin: %(funcNames), %(process)d, %(thread)d

    """
    # additional options for format:
    format_a = '%(asctime)s - %(levelname)s - %(filename)s/%(funcName)s: %(message)s'
    format_b = '%(asctime)s - %(levelname)s: %(message)s'
    logging.basicConfig(format = format_a, level = logging.DEBUG,
                        datefmt = "%d.%m.%Y %H:%M:%S",
                        handlers = [
                            logging.FileHandler(file_name, mode = file_mode),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger()

    return logger


def improve_pandas_viewing_options():
    import pandas as pd
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.min_rows', 20)
    pd.set_option('display.max_rows', 100)


# %%  Project-specific utilities


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
        triples_df, property_encoding_ID = (None, None)
        print(f"Dataset name {name_of_dataset_processed} not found!")

    return triples_df, property_encoding_ID


def add_PID_labels_as_dfcolumn():
    # refer to Wikdiata_relation_counts.py from line 366
    raise NotImplementedError()


def add_QID_labels_as_dfcolumn():
    # refer to Wikdiata_relation_counts.py from line 366
    raise NotImplementedError()


def create_train_val_test_split_from_single_TSV(train_val_test_split: tuple = (0.8, 0.1, 0.1),
                                                random_state: int = 42,
                                                rel_path_to_human_facts_file: str = 'data_preprocessing/wikidata5m_human_facts_subset_complete_040122.tsv'):
    """
    Use this to create train,validation, test files that are read
    by the Wikidata5M_pykeen class.

    This happens completely outside of training for reproducibility purposes!
    Put this in a meaningful folder that is part of the final repo

    :return: nothing, all 3 splits are saved to disk as TSV files
    """
    BASE_PATH_HOST = set_base_path_based_on_host()
    absolute_path = os.path.join(BASE_PATH_HOST, rel_path_to_human_facts_file)

    # make tests for expected input
    assert type(train_val_test_split) == tuple
    assert len(train_val_test_split) == 3, "You need to specify exactly 3 values!"
    assert sum(train_val_test_split) == 1, "Values need to add up to 1!"

    tf = TriplesFactory.from_path(absolute_path)
    train, validation, test = tf.split(list(train_val_test_split), random_state = random_state,
                                       method = 'coverage')  # coverage is default, other: cleanup

    file_name_train = 'data_preprocessing/' + f'training_data_{train_val_test_split[0]}_rs{random_state}_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.tsv'
    file_name_validation = 'data_preprocessing/' + f'validation_data_{train_val_test_split[1]}_rs{random_state}_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.tsv'
    file_name_test = 'data_preprocessing/' + f'test_data_{train_val_test_split[2]}_rs{random_state}_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.tsv'

    # save the numpy arrays (bla.triples) as TSV to disk
    np.savetxt(fname = file_name_train, X = train.triples, fmt = '%s', delimiter = '\t')
    np.savetxt(fname = file_name_validation, X = validation.triples, fmt = '%s', delimiter = '\t')
    np.savetxt(fname = file_name_test, X = test.triples, fmt = '%s', delimiter = '\t')


def filter_pykeen_results_file(path, filter_str: list, save: bool = True,
                               format: str = 'csv'):
    """

    Parameters
    ----------
    path: absolute path to file
    filter_str: string or list of strings, will be used to filter column 'key'
    e.g. ['both', 'realistic']
    format: string for determining format, can be csv or json

    Returns
    -------

    """
    if format == 'csv':
        all_results = pd.read_csv(path)
    if format == 'json':
        # TODO figure out how to do this for JSON files
        raise NotImplementedError()

    assert all(all_results.columns == pd.Index(['type', 'step', 'key', 'value'])), "Unknown dataframe format! Check the path."

    # filter for results
    filtered_results = all_results
    # TODO maybe change this to a regular expresssion instead?
    for i in filter_str:
        filter_str = filtered_results['key'].str.contains(i)
        filtered_results = filtered_results[filter_str]

    # make 'value' column numeric and round it to two digits
    filtered_results['value'] = filtered_results['value'].astype(float).apply(lambda x: round(x, 2))

    if save:
        os.chdir(os.path.dirname(path))
        print(f' Saving filtered dataframe to: {os.getcwd()}')
        filtered_results.to_csv(os.path.join(os.getcwd(), 'filtered_pykeen_results.csv'))

    return filtered_results

