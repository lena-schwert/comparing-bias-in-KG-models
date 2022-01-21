# in-built modules
import argparse
import os
import logging
import socket
import json
import shutil
from datetime import datetime

# TODO hacky fix: append project path to PYTHONPATH such that I can import from my utils.py
import sys

if socket.gethostname() == 'Schlepptop':
    path_to_append = '/home/lena/git/master_thesis_bias_in_NLP/'
# covers all CPU + GPU nodes of the HPI
elif 'node' in socket.gethostname() or socket.gethostname() in ['a6k5-01', 'dgxa100-01', 'ac922-01',
                                                                'ac922-02']:
    path_to_append = '/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis'
else:
    path_to_append = None
    ValueError("Host name is not recognized!")
sys.path.append(path_to_append)

# 3rd party custom modules
import pandas as pd
import torch
import pykeen
from pykeen.datasets.base import PathDataset
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.trackers import CSVResultTracker, TensorBoardResultTracker
from pykeen.evaluation import RankBasedEvaluator
from pykeen.utils import set_random_seed, is_cudnn_error, is_cuda_oom_error, check_shapes

# imports from my own code
from utils import set_base_path_based_on_host, initialize_my_logger

BASE_PATH_HOST = set_base_path_based_on_host()

# %% Create argument parser and all its default values

# TODO set defaults for all parameters here!
parser = argparse.ArgumentParser(
    description = 'Run experiments with Wikidata5M as the data source and knowledge graph embeddings (pykeen) as the model(s).')
parser.add_argument('-n', '--name', type = str,
                    help = 'Experiment name, a folder with this name will be created in the results folder.')
parser.add_argument('--kge', type = list,
                    help = 'Provide a list [...] of strings, where each string must match a pykeen model name.')
parser.add_argument('-d', '--debug', action = 'store_true',
                    help = 'Debugging or not? If this flag is added, this modifies the folder name and a subset of Wikidata5M is used.')
parser.add_argument('--debug-nations', action = 'store_true',
                    help = 'Use this flag when you want to use the pykeen Nations dataset. Useful for fast runs when developing the code.')
parser.add_argument('--rs', type = int, default = 42,
                    help = 'Set the random seed that will be used for all libraries.')
parser.add_argument('-e', '--epochs', type = int, default = 1000,
                    help = 'Number of epochs the model should be trained for.')
parser.add_argument('--trainset', type = str,
                    default = 'data_preprocessing/training_data_0.8_rs42_04_01_2022_19:13.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')
parser.add_argument('--valset', type = str,
                    default = 'data_preprocessing/validation_data_0.8_rs42_04_01_2022_19:13.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')
parser.add_argument('--testset', type = str,
                    default = 'data_preprocessing/test_data_0.8_rs42_04_01_2022_19:13.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')

args = parser.parse_args()


# %% Create pykeen dataset from W5M human subgraph split files


class Wikidata5M_pykeen(PathDataset):
    """
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
        self.name = "Wikidata5M"

        # TODO load existing relation/entity to ID mapping from disk

        # these are default, required attributes implemented by PathDataset
        self.training_path = os.path.join(base_path_host, rel_training_set_path)
        self.validation_path = os.path.join(base_path_host, rel_validation_set_path)
        self.testing_path = os.path.join(base_path_host, rel_test_set_path)
        self.create_inverse_triples = False
        self.load_triples_kwargs = None
        # after this line, pykeen actually creates the dataset!
        logging.debug('Dataset has been created.')

        # TODO unclear why this is really necessary or whether it could be omitted...
        # this makes all attributes from the PathDataset class accessible
        # to my Wikidata5M_pykeen class (adopted from Keidar repo
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

        # TODO save entity/relation to ID mapping (dict) to disk for reproducibility
        # with

        self.entity_to_id
        self.relation_to_id


# %% create end-to-end function for training KG only model


def run_pykeen_hpo_pipeline(KGE_MODEL_NAME_S: list = args.kge, EXPERIMENT_NAME: str = args.name,
                            DEBUGGING: bool = args.debug,
                            DEBUG_WITH_NATIONS: bool = args.debug_nations,
                            RANDOM_SEED: int = args.rs, NUM_EPOCHS: int = args.epochs,
                            rel_training_set_path = args.trainset,
                            rel_validation_set_path = args.valset,
                            rel_test_set_path = args.testset):
    """

    Parameters
    ----------

    """
    # useful assert statements
    assert EXPERIMENT_NAME is not None
    assert KGE_MODEL_NAME_S is not None
    # TODO add more?
    print(args)

    # set dataset paths and experiment name
    if DEBUGGING:
        EXPERIMENT_NAME = START_TIME + '_DEBUGGING_' + EXPERIMENT_NAME
        rel_training_set_path = 'data_preprocessing/training_data_0.8_rs42_06_01_2022_15:58_DEBUGGING.tsv'
        rel_validation_set_path = 'data_preprocessing/validation_data_0.1_rs42_06_01_2022_15:58_DEBUGGING.tsv'
        rel_test_set_path = 'data_preprocessing/test_data_0.1_rs42_06_01_2022_15:58_DEBUGGING.tsv'
    else:
        EXPERIMENT_NAME = START_TIME + '_' + EXPERIMENT_NAME
        rel_training_set_path = rel_training_set_path
        rel_validation_set_path = rel_validation_set_path
        rel_test_set_path = rel_test_set_path
    # create path for saving all the result files
    DIRECTORY_FOR_SAVING = os.path.join(BASE_PATH_HOST, 'results', EXPERIMENT_NAME)

    # create directory and then use it as working directory
    if not os.path.isdir(DIRECTORY_FOR_SAVING):
        os.mkdir(DIRECTORY_FOR_SAVING)
    os.chdir(DIRECTORY_FOR_SAVING)

    # save the currently running script file for later reference
    file_name = 'script_' + EXPERIMENT_NAME + '.py'
    shutil.copy(src = __file__, dst = os.path.join(os.getcwd(), file_name))

    # save the

    # create a logger
    logger = initialize_my_logger(file_name = EXPERIMENT_NAME + '.log')

    logger.info('The experiment has started!')
    logger.info(f'Running experiment on host: {socket.gethostname()}')
    if DEBUGGING:
        logger.info('DEBUGGING MODE: Utilizing a small subset of the data.')
    logger.info('################# MODELS #################')
    logger.info(f'Model(s): {KGE_MODEL_NAME_S}')
    logger.info('################# CUDA #################')
    logger.info(f"CUDA is used: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f'CUDA detects {torch.cuda.device_count()} device(s).')
        logger.info(f'CUDA device currently used: {torch.cuda.current_device()}')

    # TODO create metadata.JSON dictionary for storing all of this functions args to disk

    metadata_saved_on_end = args.__dict__
    # merge all dictionary objects I want to save
    dict_bla = dict() | dict()

    # try and save args namespace in json format
    with open('commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent = 2)
    f.close()

    # create a dataset instance
    if DEBUG_WITH_NATIONS:
        dataset = 'Nations'
    else:
        dataset = Wikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                    rel_training_set_path = rel_training_set_path,
                                    rel_validation_set_path = rel_validation_set_path,
                                    rel_test_set_path = rel_test_set_path)

    # set random seed for python, numpy and torch
    set_random_seed(RANDOM_SEED)

    # set device as GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('################# TRAINING #################')
    logger.info("Starting training with pykeen pipeline...")
    # do one hpo_pipeline for each model
    for model in KGE_MODEL_NAME_S:
        logger.info(f'Now training model: {model}')
        pipeline_result = pipeline(  # automatic_memory_optimization = True,
            dataset = dataset,  # TODO this is usually Wikidata5M_pykeen
            device = device,  # automatically determine
            # epochs = 1000,  # PLACEHOLDER ######################
            evaluator = RankBasedEvaluator(filtered = True),
            # default: enableBordes 2013 filtered setting
            # evaluation_relation_whitelist = {'P21'},  # useful when evaluating only specific relations
            # filter_validation_when_testing = True,
            # loss = None,  # PLACEHOLDER ######################
            # loss_kwargs = None,   # PLACEHOLDER ######################
            # lr_scheduler = None,   # PLACEHOLDER ######################
            # metadata = {}  ,  #  TODO A JSON dictionary to store with the experiment details/parameters
            # metric = None,  # PLACEHOLDER ######################
            model = model,  # specify KGE model name (as string)
            model_kwargs = dict(  # model parameters for instantiating
                embedding_dim = 100),  # model_kwargs_ranges = None,  # to override HPO defaults
            negative_sampler = 'basic',  # PLACEHOLDER ######################
            # provide string or pykeen.sampling class
            # negative_sampler_kwargs = None,  # PLACEHOLDER ######################
            # negative_sampler_kwargs_ranges = None,  # PLACEHOLDER ######################
            # optimizer = None,  # PLACEHOLDER ######################
            # optimizer_kwargs = dict(lr = 0.001),
            random_seed = RANDOM_SEED,  # regularizer = None,  # PLACEHOLDER ######################
            # regularizer_kwargs = None,  # PLACEHOLDER ######################
            result_tracker = CSVResultTracker,  # supply either pykeen.trackers class or string
            result_tracker_kwargs = dict(
                # experiment_path = DIRECTORY_FOR_SAVING,  # only applicable to TensorBoardResultTracker
                path = os.path.join(BASE_PATH_HOST, 'results', EXPERIMENT_NAME,
                                    f'pykeen_params_metrics_{EXPERIMENT_NAME}_{START_TIME}.csv'),
                # name = 'bla.csv'  # optional give custom name instead of timestamp
            ),  # stopper = 'early',  # PLACEHOLDER ######################
            # training_loop = None,  # PLACEHOLDER ######################
            training_kwargs = dict(num_epochs = NUM_EPOCHS,  # batch_size = None,
                                   checkpoint_name = 'checkpoint_' + EXPERIMENT_NAME + '.pt',
                                   checkpoint_frequency = 30,
                                   # After how many minutes should a checkpoint be saved?
                                   checkpoint_directory = os.path.join(BASE_PATH_HOST, 'results',
                                                                       EXPERIMENT_NAME,
                                                                       'checkpoints'),
                                   checkpoint_on_failure = True
                                   # to save a checkpoint when training loop fails
                                   ))

        logger.info('################# TRAINING FINISHED #################')
        logger.info('Now saving results to disk....')
        # TODO save all results to disk
        # saves trained_model.pickle, results.json, metadata.json
        pipeline_result.save_to_directory(DIRECTORY_FOR_SAVING, save_metadata = True)

        # save training and evaluation time (int)
        pipeline_result.train_seconds
        pipeline_result.evaluate_seconds

        # save losses manually (list of floats)
        pipeline_result.losses

        # save metadata JSON file manually (dict)
        pipeline_result.metadata

        # save metric results from evaluation
        pipeline_result.metric_results

        # save the model manually
        pipeline_result.save_model('trained_model_man.pkl')








        # TODO save entity and relation to ID mapping

        # TODO save triples_factory needed for link prediction
        # save pipeline_result.training

        logger.info('Done saving results!')

    # return pipeline_result


def analyze_results_from_hpo_pipeline(pipeline_results):
    pipeline_results.plot_losses()
    pipeline_results.plot()
    pass


def main():
    """



    Returns
    -------

    """

    # call either a single model pipeline or HPO


# %% Main code

if __name__ == '__main__':
    START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")
    # run_pykeen_hpo_pipeline()
    run_pykeen_hpo_pipeline(DEBUGGING = True, DEBUG_WITH_NATIONS = True, KGE_MODEL_NAME_S = ['TransE'],
                            EXPERIMENT_NAME = 'developing_code', NUM_EPOCHS = 80)
    END_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")
    logging.info(f'The experiment has ended!')
