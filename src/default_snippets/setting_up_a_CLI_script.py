### Imports

# in-built modules
import argparse
import logging
import os
import random
import shutil
import socket
from datetime import datetime

# imports from installed libraries
import numpy as np
import torch

# imports from my own code
from src.utils import set_base_path_based_on_host, initialize_my_logger, \
    improve_pandas_viewing_options, save_argparse_obj_to_disk

BASE_PATH_HOST = set_base_path_based_on_host()
improve_pandas_viewing_options()

# makes it compatible with logging coming from other sources
logger = logging.getLogger(__name__)

START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")

parser = argparse.ArgumentParser(
    description = 'Calculate bias on Wikidata5M using a trained link prediction model')

parser.add_argument('--model_type', required = True, choices = {'KG_only', 'KG+LM', 'LM_only'},
                    help = 'Specify the model type that is used.')
parser.add_argument('-n', '--name', type = str,
                    help = 'Experiment name. A folder with this name will be created  or loaded '
                           'from in the results/bias_measurement/link_prediction_bias folder.')
parser.add_argument("--data_dir",
                    default = 'data/processed/human_Wikidata5M',
                    type = str,
                    help = "The input data dir. Should contain the .tsv files (or other data files) for the task.")

parser.add_argument('--epochs', type = int,
                    help = "Number of training epochs of link prediction classifier (used for DP & PP), default to 100",
                    default = 100)
parser.add_argument('--batch', type = int,
                    help = "Training batch size for the target relation classifier.", default = 256)
parser.add_argument('--clsf_type', type = str, choices = {'mlp', 'rf'},
                    help = "Type of target relation classifier.", default = 'mlp')
parser.add_argument('--num_classes', type = int,
                    help = "Number of training epochs of link prediction classifier (used for DP & PP), default to 100",
                    default = 11)

parser.add_argument('--embedding_name', type = str, default = 'trans_e', help = "Only valid for KG_only. Embedding name, must be one of complex, conv_e, distmult, rotate, trans_d, trans_e. \
                            Default to trans_e")
parser.add_argument('--embedding_path', type = str, help = "Specify a full path to your trained embedding model. It will override default path \
                          inferred by dataset and embedding")
parser.add_argument('--predictions_path', type = str, help = 'path to predictions used in parity distance, specifying \
                           it will override internal inferred path')

parser.add_argument('--use-pretrained', action = 'store_true', help = "")
parser.add_argument('--dataset_name', type = str, default = 'Wikidata5M',
                    choices = {'Wikidata5M', 'FB15K-237'},
                    help = "Dataset name, must be one of Wikidata5M or FB15K-237. Defaults to Wikidata5M.")
parser.add_argument('-d', "--debug", action = 'store_true',
                    help = "Add this flag when debugging.")
parser.add_argument("--random_seed", default = 42, type = int,
                    help = "Integer specifying a random seed")

args = parser.parse_args()

if args.debug:
    EXPERIMENT_NAME = START_TIME + '_' + args.name
    # test + dev have ... examples, training has ...
    args.data_dir = os.path.join(args.data_dir, 'for_debugging')
    args.epochs = 10
else:
    EXPERIMENT_NAME = 'DEBUGGING_' + START_TIME + '_' + args.name

# Check that the data_dir exists and that it contains the necessary files
args.data_dir = os.path.join(BASE_PATH_HOST, args.data_dir)
if not os.path.exists(args.data_dir):
    raise FileNotFoundError('The data directory does not exist!')
list_of_necessary_files = ['train.tsv', 'validation.tsv', 'test.tsv']
for file_name in list_of_necessary_files:
    assert os.path.isfile(os.path.join(args.data_dir, file_name)), \
        f'File {file_name} does not exist in the specified data_dir!'


DIRECTORY_FOR_SAVING_OR_LOADING = os.path.join(BASE_PATH_HOST,
                                               'results/bias_measurement/link_prediction_bias',
                                               EXPERIMENT_NAME)

# create directory and then use it as working directory
if os.path.exists(DIRECTORY_FOR_SAVING_OR_LOADING) and os.listdir(
        DIRECTORY_FOR_SAVING_OR_LOADING) and args.do_train:
    raise ValueError("Output directory ({}) already exists and is not empty.".format(
        DIRECTORY_FOR_SAVING_OR_LOADING))
if not os.path.exists(DIRECTORY_FOR_SAVING_OR_LOADING):
    os.makedirs(DIRECTORY_FOR_SAVING_OR_LOADING)

    # change working directory to respective folder
    os.chdir(DIRECTORY_FOR_SAVING_OR_LOADING)

# save the currently running script file for later reference
file_name_script = 'script_' + EXPERIMENT_NAME + '.py'
source_path = os.path.join(BASE_PATH_HOST, 'src/CLI_scripts', __file__)
shutil.copy(src = source_path, dst = os.path.join(os.getcwd(), file_name_script))

# save the argparse arguments to disk
save_argparse_obj_to_disk(argparse_namespace = args)

# initialize logger
logger_file_name = f'log_LP_bias_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'
logger = initialize_my_logger(file_name = logger_file_name, level = logging.DEBUG)
logger.info(f'Saving everything in folder: {DIRECTORY_FOR_SAVING_OR_LOADING}')

if args.debug:
    logger.info(f'DEBUGGING MODE: Using a very small subset of {args.dataset_name}.')

logger.info('################# DEVICE INFORMATION #################')
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
logger.info(f'Device used: {device}')
logger.info(f'CUDA is used: {torch.cuda.is_available()}"')
logger.info(f'Using {torch.cuda.device_count()} device(s).')

# set the random seeds
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
