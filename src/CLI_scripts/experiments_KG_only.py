# in-built modules
import argparse
import csv
import os
import logging
import socket
import json
import time
import shutil
from datetime import datetime

# 3rd party custom modules
import torch
import pykeen
from pykeen.datasets.base import PathDataset
from pykeen.datasets import FB15k237
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.trackers import CSVResultTracker, TensorBoardResultTracker, MultiResultTracker, \
    resolve_result_trackers
from pykeen.evaluation import RankBasedEvaluator
from pykeen.utils import set_random_seed
from pykeen.models import TransE, DistMult, ComplEx, RotatE, predict
from pykeen.pipeline import PipelineResult
from pykeen.regularizers import LpRegularizer
from pykeen.losses import NSSALoss

# imports from my own code
from src.utils import set_base_path_based_on_host, initialize_my_logger, save_argparse_obj_to_disk, \
    HumanWikidata5M_pykeen

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
                    default = 'data/processed/files_per_model/KG_only_KGE/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')
parser.add_argument('--valset', type = str,
                    default = 'data/processed/files_per_model/KG_only_KGE/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')
parser.add_argument('--testset', type = str,
                    default = 'data/processed/files_per_model/KG_only_KGE/test_data_subset_0.05_rs42_06_05_2022_15:11.tsv',
                    help = 'Path relative to project directory to a TSV file with three columns: head, relation, tail.')

args = parser.parse_args()


# %% create end-to-end function for training KG only model


def run_pykeen_pipeline(KGE_MODEL_NAME_S: str = args.kge, EXPERIMENT_NAME: str = args.name,
                        DEBUGGING: bool = args.debug, DEBUG_WITH_NATIONS: bool = args.debug_nations,
                        EMBEDDING_DIM: int = args.dim, BATCH_SIZE: int = args.batch_size,
                        LEARNING_RATE: float = args.lr, RANDOM_SEED: int = args.rs,
                        NUM_EPOCHS: int = args.epochs, NEGATIVE_SAMPLES: int = args.ns,
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
        rel_training_set_path = 'data/processed/files_per_model/KG_only_KGE/for_debugging/training_data_subset_0.9_rs42_06_05_2022_15:11_for_debugging.tsv'
        rel_validation_set_path = 'data/processed/files_per_model/KG_only_KGE/for_debugging/validation_data_subset_0.05_rs42_06_05_2022_15:11_for_debugging.tsv'
        rel_test_set_path = 'data/processed/files_per_model/KG_only_KGE/for_debugging/test_data_subset_0.05_rs42_06_05_2022_15:11_for_debugging.tsv'
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
    # do one pipeline for each model
    for model in KGE_MODEL_NAME_S:
        logger.info('################# SET PARAMETERS BASED ON MODEL #################')
        logger.info(f'Now training model: {model}')

        if model == 'TransE':
            # defaults: LR = 0.001, BS = 2048, epochs = 1000, NS = 64, DIM = 512,
            # optimizer: SGD, weight decay = 0, NSSALoss(margin 12, adv. temp 0.5)
            loss_function = NSSALoss(margin = 12, adversarial_temperature = 0.5)
            regularizer = None
        elif model == 'DistMult':
            # defaults: LR = 0.1, BS = 2048, epochs = 2000, NS = 64, DIM = 512,
            # optimizer: SGD, weight decay = 0, NSSALoss(margin 0, adv. temp 2)
            loss_function = NSSALoss(margin = 0, adversarial_temperature = 2)
            regularizer = LpRegularizer(weight = 2e-3, p = 3)
        elif model == 'ComplEx':
            # defaults: LR = 0.1, BS = 2048, epochs = 1000, NS = 64, DIM = 512,
            # optimizer: SGD, weight decay = 0, NSSALoss(margin 0, adv. temp 0.2)
            loss_function = NSSALoss(margin = 0, adversarial_temperature = 0.2)
            regularizer = LpRegularizer(weight = 2e-3, p = 3)
        elif model == 'RotatE':
            # defaults: LR = 0.01, BS = 2048, epochs = 1000, NS = 64, DIM = 512,
            # optimizer: SGD, weight decay = 0, NSSALoss(margin 6, adv. temp 0.2)
            loss_function = NSSALoss(margin = 6, adversarial_temperature = 0.2)
            regularizer = None

        optimizer = 'adam'

        metadata_for_HPO = dict(
            model = model,
            folder_name = EXPERIMENT_NAME,
            batch_size = BATCH_SIZE,
            epochs = NUM_EPOCHS,
            learning_rate = LEARNING_RATE,
            dimension = EMBEDDING_DIM,
            negative_samples = NEGATIVE_SAMPLES,
            loss_function = str(loss_function).strip('()'),
            loss_params = f'margin: {loss_function.margin}, adv. temp: {loss_function.inverse_softmax_temperature}',
            regularizer = str(regularizer).strip('()') if regularizer else 'None',
            regularizer_params = f'p: {regularizer.p}, weight: {round(regularizer.weight.item(), 2)}' if regularizer else 'None',
            optimizer = optimizer
        )

        logger.info("Starting training with pykeen pipeline...")
        pipeline_result = pipeline(
            dataset = dataset,  # IMPORTANT this is usually HumanWikidata5M_pykeen
            device = device,  # device used with PyTorch: CUDA or CPU
            evaluator = RankBasedEvaluator(),
            # default: enable Bordes 2013 filtered setting
            evaluator_kwargs = dict(filtered = True, mode = 'validation',
                                    additional_filter_triples = [
                                        dataset.training.mapped_triples,
                                        dataset.validation.mapped_triples,
                                        dataset.testing.mapped_triples,
                                    ]),
            filter_validation_when_testing = True,
            loss = loss_function,  # PLACEHOLDER ######################
            metadata = metadata_for_HPO,  # information about parameters for HPO
            model = model,  # specify KGE model name (as string)
            model_kwargs = dict(  # model parameters for instantiating
                embedding_dim = EMBEDDING_DIM),
            # provide string or pykeen.sampling class
            negative_sampler = 'basic',  # basic sampler does sampling with uniform probability
            negative_sampler_kwargs = dict(num_negs_per_pos = NEGATIVE_SAMPLES),  # default: 1
            optimizer = optimizer,  # PLACEHOLDER ######################
            optimizer_kwargs = dict(lr = LEARNING_RATE),  # same as KEPLER for TransE
            random_seed = RANDOM_SEED,  # regularizer = None,  # PLACEHOLDER ######################
            regularizer = regularizer,  # PLACEHOLDER ######################
            result_tracker = 'tensorboard',  # supply either pykeen.trackers class or string
            result_tracker_kwargs = dict(experiment_path = DIRECTORY_FOR_SAVING,
                                         # only applicable to TensorBoardResultTracker
                                         ),
            # stopper = 'early',  # PLACEHOLDER ######################
            training_kwargs = dict(num_epochs = NUM_EPOCHS, batch_size = BATCH_SIZE,
                                   checkpoint_name = 'checkpoint_' + EXPERIMENT_NAME + '.pt',
                                   # After how many minutes should a checkpoint be saved?
                                   checkpoint_frequency = 10,
                                   checkpoint_directory = os.path.join(BASE_PATH_HOST,
                                                                       'results/KG_only',
                                                                       EXPERIMENT_NAME,
                                                                       'checkpoints'),
                                   # to save a checkpoint when training loop fails
                                   checkpoint_on_failure = True))

        logger.info('################# TRAINING FINISHED #################')
        logger.info('Now saving results to disk....')
        # TODO save all results to disk
        # saves trained_model.pickle, results.json, metadata.json
        pipeline_result.save_to_directory(DIRECTORY_FOR_SAVING, save_metadata = True)

        # IMPORTANT finalize the model putput using metadata.json
        filtered_result_dict = dict(
            mean_rank_both = int(pipeline_result.metric_results.get_metric('both.realistic.arithmetic_mean_rank')),
            MRR_both = round(pipeline_result.metric_results.get_metric('both.realistic.inverse_harmonic_mean_rank'), 4),
            hits_at_1_both = round(pipeline_result.metric_results.get_metric('both.realistic.hits_at_1'), 4),
            hits_at_3_both = round(pipeline_result.metric_results.get_metric('both.realistic.hits_at_3'), 4),
            hits_at_10_both = round(pipeline_result.metric_results.get_metric('both.realistic.hits_at_10'), 4),
            mean_rank_head = int(pipeline_result.metric_results.get_metric('head.realistic.arithmetic_mean_rank')),
            MRR_head = round(pipeline_result.metric_results.get_metric('head.realistic.inverse_harmonic_mean_rank'), 4),
            hits_at_1_head = round(pipeline_result.metric_results.get_metric('head.realistic.hits_at_1'), 4),
            hits_at_3_head = round(pipeline_result.metric_results.get_metric('head.realistic.hits_at_3'), 4),
            hits_at_10_head = round(pipeline_result.metric_results.get_metric('head.realistic.hits_at_10'), 4),
            mean_rank_tail = int(pipeline_result.metric_results.get_metric('tail.realistic.arithmetic_mean_rank')),
            MRR_tail = round(pipeline_result.metric_results.get_metric('tail.realistic.inverse_harmonic_mean_rank'), 4),
            hits_at_1_tail = round(pipeline_result.metric_results.get_metric('tail.realistic.hits_at_1'), 4),
            hits_at_3_tail = round(pipeline_result.metric_results.get_metric('tail.realistic.hits_at_3'), 4),
            hits_at_10_tail = round(pipeline_result.metric_results.get_metric('tail.realistic.hits_at_10'), 4),
            train_time_h = round((pipeline_result.train_seconds / 60 / 60), 2),
            eval_time_min = round((pipeline_result.evaluate_seconds / 60), 2)
        )

        metadata_for_HPO.update(filtered_result_dict)

        with open(os.path.join(DIRECTORY_FOR_SAVING, 'params_filtered_results.tsv'), 'w') as f:
            dict_writer = csv.DictWriter(f, metadata_for_HPO.keys(), delimiter = '\t')
            dict_writer.writeheader()
            dict_writer.writerow(metadata_for_HPO)

        # print training and evaluation time (int)
        logger.info(f'Training time in seconds: {pipeline_result.train_seconds}')
        logger.info(f'Evaluation time in seconds: {pipeline_result.evaluate_seconds}')

        logger.info('Done saving results!')

    return pipeline_result


def get_all_predictions_df(path_to_trained_model: str, model_class: str,
                           pipeline_result_object: PipelineResult = None):
    """

    Parameters
    ----------
    pipeline_results
    called

    Returns
    -------

    """

    # TODO implement that the predictions are only done for the selected relations!!!
    # bias + target relations!

    if pipeline_result_object:
        assert type(
            pipeline_result_object) == PipelineResult, 'Something went wrong, object is no PipelineResult!'
        trained_model = pipeline_result_object.model
        all_predictions_df = predict.get_all_prediction_df(trained_model,
                                                           triples_factory = pipeline_result_object.training)

    if not pipeline_result_object:
        if os.path.basename(path_to_trained_model) == 'trained_model.pkl':
            trained_model = torch.load(path_to_trained_model)
            # retrieve the training Triples
            dataset_HumanWikidata5M = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                                             rel_training_set_path = args.trainset,
                                                             rel_validation_set_path = args.valset,
                                                             rel_test_set_path = args.testset)
            all_predictions_df = trained_model.get_all_prediction_df(
                triples_factory = dataset_HumanWikidata5M.training)

            # TODO create the all_predictions_df

        elif 'checkpoint' in os.path.basename(path_to_trained_model):
            assert model_class in ['TransE', 'ComplEx', 'DistMult', 'RotatE']
            absolute_path = os.path.join(BASE_PATH_HOST, 'results/KG_only', path_to_trained_model)
            loaded_checkpoint = torch.load(absolute_path)

            train_triples_factory = TriplesFactory.from_path(
                path = os.path.join(BASE_PATH_HOST, args.trainset),
                entity_to_id = loaded_checkpoint['entity_to_id_dict'],
                relation_to_id = loaded_checkpoint['relation_to_id_dict'])

            if model_class == 'TransE':
                trained_model = TransE(triples_factory = train_triples_factory)
            elif model_class == 'DistMult':
                trained_model = DistMult(triples_factory = train_triples_factory)
            elif model_class == 'ComplEx':
                trained_model = ComplEx(triples_factory = train_triples_factory)
            elif model_class == 'RotatE':
                trained_model = RotatE(triples_factory = train_triples_factory)

            trained_model.load_state_dict(loaded_checkpoint['model_state_dict'])

    # TODO create the all_predictions_df

    # save the all_predictions_df whereever it was created from
    all_predictions_df.to_csv('test.')

    # pipeline_results.plot_losses()
    # pipeline_results.plot()
    pass


# %% Main code

if __name__ == '__main__':
    START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")
    experiment_start = time.perf_counter()
    pipeline_result = run_pykeen_pipeline()
    # if args.evaluate:
    #     logging.info('Evaluating the model after training is finished.')
    #     pass
    experiment_end = time.perf_counter()
    logging.info(f'The experiment has ended!')
    logging.info(f'It ran for {round((experiment_end - experiment_start) / 60, 2)} minutes.')
