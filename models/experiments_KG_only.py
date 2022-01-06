import os
import pandas as pd

import torch
import pykeen
from pykeen.datasets.base import PathDataset
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.utils import set_random_seed, is_cudnn_error, is_cuda_oom_error, check_shapes

from utils import set_base_path_based_on_host

BASE_PATH_HOST = set_base_path_based_on_host()


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

        # these are default, required attributes implemented by PathDataset
        self.training_path = os.path.join(base_path_host, rel_training_set_path)
        self.validation_path = os.path.join(base_path_host, rel_validation_set_path)
        self.testing_path = os.path.join(base_path_host, rel_test_set_path)
        self.create_inverse_triples = False
        self.load_triples_kwargs = None
        # after this line, pykeen actually creates the dataset!
        # TODO show progress bar when the dataset is created by pykeen

        # TODO unclear why this is really necessary or whether it could be omitted...
        # this makes all attributes from the PathDataset class accessible
        # to my Wikidata5M_pykeen class (adopted from Keidar repo
        # necessary to pass PathDataset's mandatory parameters here like this
        # needs to be here, because self.bla arguments need to be created first
        super().__init__(training_path = self.training_path, testing_path = self.validation_path,
                         validation_path = self.testing_path, **kwargs)

        # simple check whether the TriplesFactories were actually created
        assert type(
            self.training) == pykeen.triples.TriplesFactory, "Training dataset has not been correctly loaded as TriplesFactory!"
        assert type(
            self.validation) == pykeen.triples.TriplesFactory, "Validation dataset has not been correctly loaded as TriplesFactory!"
        assert type(
            self.testing) == pykeen.triples.TriplesFactory, "Testing dataset has not been correctly loaded as TriplesFactory!"


# %% create end-to-end function for training KG only model


def run_pykeen_hpo_pipeline(KGE_MODEL_NAME_S: list, EXPERIMENT_NAME: str,
                            DEBUGGING: bool = True, RANDOM_SEED: int = 42,
                            rel_training_set_path = 'data_preprocessing/training_data_0.8_rs42_04_01_2022_19:13.tsv',
                            rel_validation_set_path = 'data_preprocessing/validation_data_0.1_rs42_04_01_2022_19:13.tsv',
                            rel_test_set_path = 'data_preprocessing/test_data_0.1_rs42_04_01_2022_19:13.tsv'):
    """

    Parameters
    ----------

    """
    print('######  MODELS ######')
    print(f'Model(s): {KGE_MODEL_NAME_S}')
    print('######  CUDA ######')
    print(f"CUDA is used: {torch.cuda.is_available()}")
    print(f'CUDA detects {torch.cuda.device_count()} device(s).')
    if torch.cuda.is_available():
        print(f'CUDA device currently used: {torch.cuda.current_device()}')

    # add debugging to distinguish debugging results folder
    if DEBUGGING:
        print("DEBUGGING MODE: Logging to debugging folder and using smaller version of the dataset.")
        EXPERIMENT_NAME = 'DEBUGGING_'
        rel_training_set_path = 'data_preprocessing/training_data_0.8_rs42_06_01_2022_15:58_DEBUGGING.tsv'
        rel_validation_set_path = 'data_preprocessing/validation_data_0.1_rs42_06_01_2022_15:58_DEBUGGING.tsv'
        rel_test_set_path = 'data_preprocessing/test_data_0.1_rs42_06_01_2022_15:58_DEBUGGING.tsv'

    else:
        EXPERIMENT_NAME = EXPERIMENT_NAME
        rel_training_set_path = rel_training_set_path,
        rel_validation_set_path = rel_validation_set_path,
        rel_test_set_path = rel_test_set_path

    # create path for saving
    DIRECTORY_FOR_SAVING = os.path.join(BASE_PATH_HOST, 'results', EXPERIMENT_NAME)

    # set random seed for python, numpy and torch
    set_random_seed(RANDOM_SEED)

    # create a dataset instance
    W5M_dataset = Wikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                    rel_training_set_path = rel_training_set_path,
                                    rel_validation_set_path = rel_validation_set_path,
                                    rel_test_set_path = rel_test_set_path)

    # TODO automatically determine device: pykeen needs string or torch.device?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # do one hpo_pipeline for each model
    for model in KGE_MODEL_NAME_S:
        pipeline_result = hpo_pipeline(dataset = W5M_dataset,  # this is usually Wikidata5M_pykeen
                                       device = device,  # automatically determine
                                       # epochs = None,  # PLACEHOLDER ######################
                                       #evaluator = None,  # specify evaluation
                                       #filter_validation_when_testing = True,
                                       #loss = None,  # PLACEHOLDER ######################
                                       #loss_kwargs = None,   # PLACEHOLDER ######################
                                       #lr_scheduler = None,   # PLACEHOLDER ######################
                                       #metric = None,  # PLACEHOLDER ######################
                                       model = model,   # specify KGE model name (as string)
                                       model_kwargs = dict(  # model parameters for instantiating
                                           embedding_dim = 100),
                                       #model_kwargs_ranges = None,  # to override HPO defaults
                                       negative_sampler = 'basic',  # PLACEHOLDER ######################
                                       # provide string or pykeen.sampling class
                                       #negative_sampler_kwargs = None,  # PLACEHOLDER ######################
                                       #negative_sampler_kwargs_ranges = None,
                                       #optimizer = None,  # PLACEHOLDER ######################
                                       #optimizer_kwargs = dict(lr = 0.001),
                                       #random_seed = RANDOM_SEED,
                                       #regularizer = None,  # PLACEHOLDER ######################
                                       #regularizer_kwargs = None,  # PLACEHOLDER ######################
                                       result_tracker = 'tensorboard',  # PLACEHOLDER ######################
                                       result_tracker_kwargs = dict(
                                           experiment_path = DIRECTORY_FOR_SAVING),
                                       #stopper = 'early',  # PLACEHOLDER ######################
                                       #training_loop = None,  # PLACEHOLDER ######################
                                       training_kwargs = dict(num_epochs = 5),
                                       # TODO manually enable/confirm Bordes 2013 filtered setting
                                       )

        # TODO save all results to disk
        # saves model.pickle
        pipeline_result.save_to_directory(DIRECTORY_FOR_SAVING)

        # TODO save entity and relation to ID mapping

    #return pipeline_result


def analyze_results_from_hpo_pipeline(pipeline_results):
    pipeline_results.plot_losses()
    pipeline_results.plot()
    pass


# %% Main code

if __name__ == '__main__':
    run_pykeen_hpo_pipeline(DEBUGGING = True, KGE_MODEL_NAME_S = ['TransE'],
                            EXPERIMENT_NAME = 'test_tensorboard')
    pass
