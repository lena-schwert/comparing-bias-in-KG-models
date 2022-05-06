# in-built modules
import argparse
import os
import logging
import socket
import json
import shutil
from datetime import datetime

# 3rd party custom modules
import torch
import pykeen
from pykeen.datasets.base import PathDataset
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.trackers import CSVResultTracker, TensorBoardResultTracker, MultiResultTracker, resolve_result_trackers
from pykeen.evaluation import RankBasedEvaluator
from pykeen.utils import set_random_seed

# imports from my own code
from src.utils import set_base_path_based_on_host, initialize_my_logger, \
    save_argparse_obj_to_disk
from src.utils_with_pykeen import HumanWikidata5M_pykeen

BASE_PATH_HOST = set_base_path_based_on_host()

# %% Create argument parser and all its default values

# TODO set defaults for all parameters here!
parser = argparse.ArgumentParser(
    description = 'Run experiments with Wikidata5M as the data source and knowledge graph embeddings (pykeen) as the model(s).')
parser.add_argument('-n', '--name', type = str, required = True,
                    help = 'Experiment name, a folder with this name will be created in the results/KG_only folder.')
parser.add_argument('--kge', nargs = '*', required = True,
                    help = 'Provide a comma, separated list of model names, where each one must match a pykeen model name.')
parser.add_argument('-d', '--debug', action = 'store_true',
                    help = 'Debugging or not? If this flag is added, this modifies the folder name and a subset of Wikidata5M is used.')
parser.add_argument('--debug-nations', action = 'store_true',
                    help = 'Use this flag when you want to use the pykeen Nations dataset. Useful for fast runs when developing the code.')


parser.add_argument('-e', '--epochs', type = int, required = True,
                    help = 'Number of epochs the model should be trained for.')
parser.add_argument('-bs', '--batch_size', type = int, required = True,
                    help = 'Batch size for training.')
parser.add_argument('-lr', type = float, required = True,
                    help = 'Learning rate. No default as this is model-specific.')
parser.add_argument('--dim', type = int, default = 512,
                    help = 'Dimension of the embeddings. Defaults to 512'
                           'following the Graphvite benchmark parameters of Wikidata5M.')
parser.add_argument('-ns', type = int, default = 64,
                    help = 'Number of negative samples per positive examples. Defaults to 64'
                           'following the Graphvite benchmark parameters of Wikidata5M.')
parser.add_argument('--rs', type = int, default = 42,
                    help = 'Set the random seed that will be used for all libraries.')

parser.add_argument('--trainset', type = str,
                    default = 'data/processed/output_adapted_to_models/KG_only_KGE/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')
parser.add_argument('--valset', type = str,
                    default = 'data/processed/output_adapted_to_models/KG_only_KGE/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')
parser.add_argument('--testset', type = str,
                    default = 'data/processed/output_adapted_to_models/KG_only_KGE/test_data_subset_0.05_rs42_06_05_2022_15:11.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')

args = parser.parse_args()


# %% create end-to-end function for training KG only model


def run_pykeen_pipeline(KGE_MODEL_NAME_S: str = args.kge,
                        EXPERIMENT_NAME: str = args.name,
                        DEBUGGING: bool = args.debug,
                        DEBUG_WITH_NATIONS: bool = args.debug_nations,
                        EMBEDDING_DIM: int = args.dim,
                        BATCH_SIZE: int = args.batch_size,
                        LEARNING_RATE: float = args.lr,
                        RANDOM_SEED: int = args.rs,
                        NUM_EPOCHS: int = args.epochs,
                        NEGATIVE_SAMPLES: int = args.ns,
                        rel_training_set_path: str = args.trainset,
                        rel_validation_set_path: str = args.valset,
                        rel_test_set_path: str = args.testset):
    """

    Parameters
    ----------

    """
    # useful assert statements
    assert EXPERIMENT_NAME is not None

    # create list of strings out of the model names: taken from pykeen/models/__init__.py
    for model in KGE_MODEL_NAME_S:
        assert model in ['TransE', 'ComplEx', 'DistMult', 'QuatE', 'SimplE', 'RotatE', 'ConvE']

    print(args)

    # set dataset paths and experiment name
    if DEBUGGING:
        EXPERIMENT_NAME = START_TIME + '_DEBUGGING_' + EXPERIMENT_NAME
        rel_training_set_path = 'data/processed/output_adapted_to_models/KG_only_KGE/for_debugging/training_data_subset_0.9_rs42_06_05_2022_15:11_for_debugging.tsv'
        rel_validation_set_path = 'data/processed/output_adapted_to_models/KG_only_KGE/for_debugging/validation_data_subset_0.05_rs42_06_05_2022_15:11_for_debugging.tsv'
        rel_test_set_path = 'data/processed/output_adapted_to_models/KG_only_KGE/for_debugging/test_data_subset_0.05_rs42_06_05_2022_15:11_for_debugging.tsv'
    else:
        EXPERIMENT_NAME = START_TIME + '_' + EXPERIMENT_NAME
        rel_training_set_path = rel_training_set_path
        rel_validation_set_path = rel_validation_set_path
        rel_test_set_path = rel_test_set_path
    # create path for saving all the result files
    DIRECTORY_FOR_SAVING = os.path.join(BASE_PATH_HOST, 'results/KG_only', EXPERIMENT_NAME)

    # create directory and then use it as working directory
    if not os.path.isdir(DIRECTORY_FOR_SAVING):
        os.mkdir(DIRECTORY_FOR_SAVING)
    os.chdir(DIRECTORY_FOR_SAVING)

    # save the currently running script file for later reference
    file_name = 'script_' + EXPERIMENT_NAME + '.py'
    source_path = os.path.join(BASE_PATH_HOST, 'src/CLI_scripts', __file__)
    shutil.copy(src = source_path, dst = os.path.join(os.getcwd(), file_name))

    # save the argparse arguments to disk
    save_argparse_obj_to_disk(argparse_namespace = args)

    # create a logger
    logger_file_name = f'log_train_{socket.gethostname()}' + EXPERIMENT_NAME + '.txt'
    logger = initialize_my_logger(file_name = logger_file_name)

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

    # create a dataset instance
    if DEBUG_WITH_NATIONS:
        dataset = 'Nations'
    else:
        dataset = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                         rel_training_set_path = rel_training_set_path,
                                         rel_validation_set_path = rel_validation_set_path,
                                         rel_test_set_path = rel_test_set_path)

    # set random seed for python, numpy and torch
    set_random_seed(RANDOM_SEED)

    # set device as GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('################# TRAINING #################')
    logger.info("Starting training with pykeen pipeline...")
    # do one pipeline for each model
    for model in KGE_MODEL_NAME_S:
        logger.info(f'Now training model: {model}')
        pipeline_result = pipeline(#automatic_memory_optimization = True,
            dataset = dataset,  # IMPORTANT this is usually HumanWikidata5M_pykeen
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
                embedding_dim = EMBEDDING_DIM),  # model_kwargs_ranges = None,  # to override HPO defaults
            negative_sampler = 'basic',  # basic sampler does sampling with uniform probability
            # provide string or pykeen.sampling class
            negative_sampler_kwargs = dict(num_negs_per_pos = NEGATIVE_SAMPLES),  # default: 1
            # negative_sampler_kwargs_ranges = None,  # for HPO ######################
            # optimizer = None,  # PLACEHOLDER ######################
            optimizer_kwargs = dict(lr = LEARNING_RATE),  # same as KEPLER for TransE
            random_seed = RANDOM_SEED,  # regularizer = None,  # PLACEHOLDER ######################
            # regularizer_kwargs = None,  # PLACEHOLDER ######################
            result_tracker = 'tensorboard',  # supply either pykeen.trackers class or string
            result_tracker_kwargs = dict(
                experiment_path = DIRECTORY_FOR_SAVING,  # only applicable to TensorBoardResultTracker
            ),
            # stopper = 'early',  # PLACEHOLDER ######################
            # training_loop = None,  # PLACEHOLDER ######################
            training_kwargs = dict(num_epochs = NUM_EPOCHS,
                                   batch_size = BATCH_SIZE,
                                   checkpoint_name = 'checkpoint_' + EXPERIMENT_NAME + '.pt',
                                   # After how many minutes should a checkpoint be saved?
                                   checkpoint_frequency = 10,
                                   checkpoint_directory = os.path.join(BASE_PATH_HOST, 'results/KG_only',
                                                                       EXPERIMENT_NAME,
                                                                       'checkpoints'),
                                   # to save a checkpoint when training loop fails
                                   checkpoint_on_failure = True
                                   ))

        logger.info('################# TRAINING FINISHED #################')
        logger.info('Now saving results to disk....')
        # TODO save all results to disk
        # saves trained_model.pickle, results.json, metadata.json
        pipeline_result.save_to_directory(DIRECTORY_FOR_SAVING, save_metadata = False)

        # TODO filter results.json (both, realistic) and save it to disk with keys = columns
        # metric_results_df = pd.DataFrame.from_dict(result_dict)
        # file_name_metrics = 'filtered_link_prediction_results.csv'
        # metric_results_df.to_csv(file_name_metrics)
        # # save losses manually (list of floats)
        # pipeline_result.losses
        # # save metric results from evaluation
        # pipeline_result.metric_results
        # # save the model manually
        # pipeline_result.save_model('trained_model_man.pkl')

        # save training and evaluation time (int)
        logger.info(f'Training time in seconds: {pipeline_result.train_seconds}')
        logger.info(f'Evaluation time in seconds: {pipeline_result.evaluate_seconds}')

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
    # run_pykeen_hpo_pipeline(DEBUGGING = False, DEBUG_WITH_NATIONS = False, KGE_MODEL_NAME_S = ['TransE'],
    #                         EXPERIMENT_NAME = 'developing_code', NUM_EPOCHS = 80)
    run_pykeen_pipeline()
    END_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")
    logging.info(f'The experiment has ended!')
