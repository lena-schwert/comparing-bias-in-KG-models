# %% Imports
# IN-built functions
import json
import logging
import os
import socket
import sys
from datetime import datetime
import argparse

import numpy as np
# 3rd part modules
import pandas as pd
import pykeen
from pykeen.triples import TriplesFactory
from pykeen.datasets.base import PathDataset


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


def initialize_my_logger(file_name, level = logging.DEBUG, file_mode: str = 'a'):
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
    # list of options for format:
    format_long = '%(asctime)s - %(levelname)s - %(filename)s/%(funcName)s: %(message)s'
    format_short = '%(asctime)s - %(levelname)s: %(message)s'
    logging.basicConfig(format = format_long, level = level,
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


def search_keys_by_val(entity_to_id_dict: dict, byVal):
    keysList = []
    itemsList = entity_to_id_dict.items()
    for item in itemsList:
        if item[1] == byVal:
            keysList.append(f'key: {item[0]}, value: {byVal})')
    return keysList


def search_val_by_key(entity_to_id_dict: dict, byKey: str):
    valList = []
    itemsList = entity_to_id_dict.items()
    for item in itemsList:
        if item[0] == byKey:
            valList.append(f'key: {item[0]}, value: {byKey})')
    return valList


def start_time():
    from datetime import datetime
    START_TIME = datetime.now()
    return START_TIME


def end_time():
    from datetime import datetime
    END_TIME = datetime.now()
    return END_TIME


def time_passed(START_TIME, END_TIME):
    result = END_TIME-START_TIME
    return


def save_argparse_obj_to_disk(argparse_namespace, path = None, file_name = None):
    '''

    Parameters
    ----------
    argparse_namespace: argparse.Namespace object created with in-built module.
    path: If None, save to working directory
    '''
    assert type(argparse_namespace) == argparse.Namespace, 'Object provided is not an argparse namespace!'
    if path is not None:
        file_name = os.path.join(path, 'argparse_arguments.json')
    else:
        file_name = 'argparse_arguments.json'

    with open(file_name, 'w') as f:
        json.dump(argparse_namespace.__dict__, f, indent = 2)
    f.close()
    print(f'Saved argparse object as JSON. File name: {file_name}')


# %%  Project-specific utilities


def get_triples_df(name_of_dataset_processed):
    """
    For each dataset, read in the official files

    Also returns property_encoding_ID, which is helpful

    :param name_of_dataset_processed: string that is the dataset name
    :return: tuple of dataframe with all triples
    """
    base_path = set_base_path_based_on_host()

    # do specific things for different datasets, if required
    if name_of_dataset_processed.lower() == 'wikidata5m':
        # file names: e.g. wikidata5m_all_triplets.txt
        # each line is a triple: Q29387131	P31	Q5 (tab-separated)
        dataset_folder = os.path.join(base_path, '../data/raw/SOTA_datasets_raw_downloads/Wikidata5M/')
        file_name = 'wikidata5m_all_triplets.txt'
        triples_df = pd.read_csv(os.path.join(dataset_folder, file_name), sep = '\t',
                                 names = ['head_entity', 'relation', 'tail_entity'])
        property_encoding_ID = 'P_ID'

    elif name_of_dataset_processed.lower() == 'wikidatasets-humans':
        dataset_folder = os.path.join(base_path, '../data/Wikidatasets_humans/')
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
        dataset_folder = os.path.join(base_path, '../data/Codex_S_M_L/triples/')
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


class HumanWikidata5M_pykeen(PathDataset):
    """
    This class is based on the file wiki5m/Wiki5m.py from https://github.com/mianzg/kgbiasdetec

    This class inherits from pykeen's PathDataset that does lazy loading.
    I provide paths to 3 existing train-validation-test split tsv files on disk
    instead of doing the split here in this class to make sure the same data is always used.
    --> Reproducibility!

    Internally, PathDataset then uses TriplesFactory.from_path() + triples.utils.load_triples()
    to create the data objects pykeen needs internally.

    This expects TSV files with three ordered columns: head, relation, tail!!!
    """

    def __init__(self, base_path_host: str, rel_training_set_path: str,
                 rel_validation_set_path: str, rel_test_set_path: str, **kwargs):
        self.name = "HumanWikidata5M"
        self.base_path_host = base_path_host
        # TODO change this to the final location in processed folder
        self.location_of_mapping_files = os.path.join(self.base_path_host, 'data/interim/')

        # these are default, required attributes implemented by PathDataset
        self.training_path = os.path.join(base_path_host, rel_training_set_path)
        self.validation_path = os.path.join(base_path_host, rel_validation_set_path)
        self.testing_path = os.path.join(base_path_host, rel_test_set_path)
        self.create_inverse_triples = False
        self.load_triples_kwargs = None
        # after this line, pykeen actually creates the dataset!
        # TripelesFactory are created: self.training, self.validation self.testing
        logging.debug('Dataset has been created.')

        # TODO unclear why this is really necessary or whether it could be omitted...
        # this makes all attributes from the PathDataset class accessible
        # to my HumanWikidata5M_pykeen class (adopted from Keidar repo
        # necessary to pass PathDataset's mandatory parameters here like this
        # needs to be down here, because self.bla arguments need to be created first
        super().__init__(training_path = self.training_path, testing_path = self.validation_path,
                         validation_path = self.testing_path, **kwargs)

        # simple check whether the TriplesFactories were actually created
        assert type(
            self.training) == pykeen.triples.TriplesFactory, "Training dataset has not been correctly loaded as TriplesFactory!"
        assert type(
            self.validation) == pykeen.triples.TriplesFactory, "Validation dataset has not been correctly loaded as TriplesFactory!"
        assert type(
            self.testing) == pykeen.triples.TriplesFactory, "Testing dataset has not been correctly loaded as TriplesFactory!"
        logging.debug('Wikidata5M passed all assertions.')

    def _load(self) -> None:
        """
        Override this pykeen method, such that the entity_to_ID label is not
        automatically created but imported from a file on disk.

        This is useful for increased reproducibility, so the mapping is not automatically created!

        Informed by options for function TriplesFactory.from_path()
        """
        # extract the first 'ID' and the second 'num ID' column from the mapping files
        # final format: {'Q100': 0},  {'P21': 0}
        with open(os.path.join(self.location_of_mapping_files, 'entity_ID_to_numID_to_label_06042022_v1.tsv'), 'r') as f1:
            entity_to_id_from_disk = {key: int(value) for key, value in (line.split()[0:2] for line in f1)}
            f1.seek(0)
            entity_numID_to_label_from_disk = {int(key): value for key, value in (line.split()[1:3] for line in f1)}
        f1.close()
        with open(os.path.join(self.location_of_mapping_files, 'relation_ID_to_numID_to_label_06042022_v1.tsv'), 'r') as f2:
            relation_to_id_from_disk = {key: int(value) for key, value in (line.split()[0:2] for line in f2)}
            f2.seek(0)
            relation_numID_to_label_from_disk = {int(key): value for key, value in
                                                 (line.split()[1:3] for line in f2)}
        f2.close()

        # all keys should start with P or Q
        assert len(entity_to_id_from_disk) == sum([True if k.startswith('Q') else False for k, v in entity_to_id_from_disk.items()])
        assert len(relation_to_id_from_disk) == sum([True if k.startswith('P') else False for k, v in relation_to_id_from_disk.items()])

        self._training = TriplesFactory.from_path(
            path=self.training_path,
            create_inverse_triples=self.create_inverse_triples,
            load_triples_kwargs=self.load_triples_kwargs,
            entity_to_id = entity_to_id_from_disk,
            relation_to_id = relation_to_id_from_disk
        )
        self._testing = TriplesFactory.from_path(
            path=self.testing_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
            # do not explicitly create inverse triples for testing; this is handled by the evaluation code
            create_inverse_triples=False,
            load_triples_kwargs=self.load_triples_kwargs,
        )

        # both mapping files should have the same numeric IDs
        assert list(entity_numID_to_label_from_disk.keys()) == list(self.entity_to_id.values())
        assert list(relation_numID_to_label_from_disk.keys()) == list(self.relation_to_id.values())

        self.entity_numID_to_label = entity_numID_to_label_from_disk
        self.relation_numID_to_label = relation_numID_to_label_from_disk



def create_train_val_test_split_from_single_TSV(train_val_test_split: tuple = (0.8, 0.1, 0.1),
                                                random_state: int = 42,
                                                rel_path_to_human_facts_file: str = 'data_preprocessing/wikidata5m_human_facts_subset_complete_040122.tsv'):
    """
    Use this to create train,validation, test files that are read
    by the HumanWikidata5M_pykeen class.

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

    file_name_train = f'data/interim/training_data_{train_val_test_split[0]}_rs{random_state}_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.tsv'
    file_name_validation = f'data/interim/validation_data_{train_val_test_split[1]}_rs{random_state}_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.tsv'
    file_name_test = f'data/interim/test_data_{train_val_test_split[2]}_rs{random_state}_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.tsv'

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
    a filtered dataframe
    """
    if format == 'csv':
        all_results = pd.read_csv(path)
    if format == 'json':
        # TODO figure out how to do this for JSON files
        raise NotImplementedError()

    assert all(all_results.columns == pd.Index(['type', 'step', 'key', 'value'])), "Unknown dataframe format! Check the path."

    # filter for results
    filtered_results = all_results
    # TODO maybe change this to a regular expression instead?
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


def find_top_k_tail_entities(relation_P_ID):
    '''
    Given a Wikidata PID, this function

    It uses the dataframe created from the human facts subset of Wikidata5M.

    Parameters
    ----------
    relation_P_ID

    Returns
    -------

    '''


def calculate_LP_metrics_from_ranks(path_to_ranks_file):
    """
    This function is intended for use with the two-column ranks files that the KG and LM model
    (i.e. KG-BERT) outputs. Sometimes it might be necessary to calculate the link prediction metrics
    independent of running the model, e.g. in cases where the run did not complete, but you want
    to get a feeling for the model performance anyway.

    Metrics implemented: mean rank

    Returns
    -------

    """
    raise NotImplementedError

    result_dict = None

    return result_dict




