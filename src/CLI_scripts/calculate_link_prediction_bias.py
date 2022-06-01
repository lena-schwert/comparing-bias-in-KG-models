# This script is based on: https://github.com/mianzg/kgbiasdetec
# More specifically, the following files:
#   experiments/run_tail_prediction.py
#   predict_tails.py

### Imports

# in-built modules
import argparse
import csv
import logging
import os
import random
import shutil
import socket
import sys
import time
from datetime import datetime
import copy
import pickle

# imports from installed libraries
import pykeen.datasets
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch

# imports from my own code
from src.utils import set_base_path_based_on_host, initialize_my_logger, \
    improve_pandas_viewing_options, save_argparse_obj_to_disk, HumanWikidata5M_pykeen
from src.bias_measurement.link_prediction_bias.predict_tails import get_preds_df
from src.bias_measurement.link_prediction_bias.BiasEvaluator import BiasEvaluator
from src.bias_measurement.link_prediction_bias.Measurement import DemographicParity, \
    PredictiveParity
from src.bias_measurement.link_prediction_bias.utils import get_sensitive_and_target_relations

BASE_PATH_HOST = set_base_path_based_on_host()
improve_pandas_viewing_options()


def get_preds_df(path_to_existing_preds_df: str = None):
    """

    Arguments needed for this are:
        dataset = a dataset object, originally inherited from pykeen.PathDataset(LazyDataset(Dataset))
        target_relation = an identifier of the target relation, will be 'P106' (occupation) for W5M
        bias_relations = list of identifier for relations that bias should be calculated for
        (originally in utils.suggest_relations: ['P27', 'P735', 'P19', 'P54', 'P69', 'P641', 'P20', 'P1344', 'P1412', 'P413'])

    Returns
    -------

    """
    if path_to_existing_preds_df:
        preds_df = pd.read_csv(path_to_existing_preds_df, sep = '\t')
        # TODO specify file names and seperator
        return preds_df

    pass


START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")

parser = argparse.ArgumentParser(
    description = 'Calculate predictive bias using a trained link prediction model')

parser.add_argument('--name', type = str,
                    help = 'Name of run, will determine folder name where results are stored')
parser.add_argument('--preds_df_path', type = str, required = True,
                    help = 'Specifies path to an existing preds_df. ')
parser.add_argument('--sensitive_df_path', type = str,  # required = True,
                    help = 'Specifies path to dataframe that informations')
parser.add_argument('--dataset_name', type = str, default = 'HumanWikidata5M',
                    choices = {'HumanWikidata5M', 'FB15K-237'},
                    help = "Dataset name, must be one of Wikidata5M or FB15K-237. Defaults to Wikidata5M.")

parser.add_argument('-d', "--debug", action = 'store_true', help = "Add this flag when debugging.")

args = parser.parse_args()

if args.debug:
    EXPERIMENT_NAME = 'DEBUGGING_' + START_TIME + '_' + args.name
    if args.dataset_name == 'Wikidata5M':
        # test has 10 examples, training has
        args.data_dir = os.path.join(args.data_dir, 'for_debugging')
    args.num_classes = 2
else:
    EXPERIMENT_NAME = START_TIME + '_' + args.name

DIRECTORY_FOR_SAVING_OR_LOADING = os.path.join(BASE_PATH_HOST,
                                               'results/bias_measurement/link_prediction_bias',
                                               EXPERIMENT_NAME)

# create directory and then use it as working directory
# if os.path.exists(DIRECTORY_FOR_SAVING_OR_LOADING) and os.listdir(
#        DIRECTORY_FOR_SAVING_OR_LOADING) and args.do_train:
#    raise ValueError(f"Output directory ({DIRECTORY_FOR_SAVING_OR_LOADING}) already exists and is not empty.")
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
logger_file_name = f'log_measure_LP_bias_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'
logger = initialize_my_logger(file_name = logger_file_name, level = logging.INFO)
logger.info(f'Saving everything in folder: {DIRECTORY_FOR_SAVING_OR_LOADING}')

if args.debug:
    logger.info(f'DEBUGGING MODE: Using a very small subset of {args.dataset_name}.')

logger.info('################# DEVICE INFORMATION #################')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device used: {device}')
logger.info(f'CUDA is used: {torch.cuda.is_available()}')
logger.info(f'Using {torch.cuda.device_count()} device(s).')

preds_df = pd.read_csv(os.path.join(BASE_PATH_HOST, args.preds_df_path), sep = '\t')

# IMPORTANT Given a preds_df, calculate bias scores using pure sklearn


# IMPORTANT Given a preds_df, calculate bias scores using fairlearn toolkit

from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

preds_df_only_valid_triples = preds_df[preds_df['P21'] != "-1"]

# multiclasss accuracy disparity
acc = MetricFrame(metrics = accuracy_score,  # sample_params = dict(average = None),
                  y_true = preds_df_only_valid_triples.true_tail_class_label,
                  y_pred = preds_df_only_valid_triples.pred_tail_class_label,
                  sensitive_features = preds_df_only_valid_triples.P21)
acc.by_group  # series with accuracy for each sensitive group
acc.difference()  # difference in accuracy between the two sensitive groups

# do the same for precision (= predictive parity)
import functools
multiclass_precision_macro = functools.partial(precision_score, average = 'macro')

prec = MetricFrame(metrics = multiclass_precision_macro,
                   y_true = preds_df_only_valid_triples.true_tail_class_label,
                   y_pred = preds_df_only_valid_triples.pred_tail_class_label,
                   sensitive_features = preds_df_only_valid_triples.P21)
prec.by_group

# do the same for recall (= equality of opportunity)
# equal TPR = equal FNR mathematically, and TPR = recall = sensitivity
multiclass_recall_macro = functools.partial(recall_score, average = 'macro')

recall = MetricFrame(metrics = multiclass_recall_macro,
                   y_true = preds_df_only_valid_triples.true_tail_class_label,
                   y_pred = preds_df_only_valid_triples.pred_tail_class_label,
                   sensitive_features = preds_df_only_valid_triples.P21)
recall.by_group

# do the same for demographic parity

demographic_parity_difference_fairlearn = demographic_parity_difference(
    y_true = preds_df_only_valid_triples.true_tail_class_label,
    y_pred = preds_df_only_valid_triples.pred_tail_class_label,
    sensitive_features = preds_df_only_valid_triples.P21, method = 'between_groups',
    # 'between_groups' computes the maximum difference between any two pairs of groups in the
    #   by_group property (i.e. group_max() - group_min())
    # 'to_overall' computes the difference between each subgroup and the corresponding value
    #   from overall (if there are control features, then overall is multivalued for each metric).
    # The result is the absolute maximum of these values.
    sample_weight = None)

# try a confusion matrix
# source: https://github.com/fairlearn/fairlearn/issues/752

# pandified confusion matrix with clarifying column annotations
def confusion_matrix_pd(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, normalize='all')
    return pd.DataFrame(matrix).rename(
               index={0: "true=0", 1: "true=1", 2: "true=2",
                      3: "true=3", 4: "true=4", 5: "true=5",
                      6: "true=6", 7: "true=7", 8: "true=8"},
               columns={0: "pred=0", 1: "pred=1", 2: "pred=2",
                      3: "pred=3", 4: "pred=4", 5: "pred=5",
                      6: "pred=6", 7: "pred=7", 8: "pred=8"}).stack()

# construct metric frame
mf = MetricFrame(metrics = confusion_matrix_pd,
                 y_true = preds_df_only_valid_triples.true_tail_class_label,
                 y_pred = preds_df_only_valid_triples.pred_tail_class_label,
                 sensitive_features = preds_df_only_valid_triples.P21)

# mf.by_group needs to be transformed to work properly
conf_matrix = pd.DataFrame(mf.by_group.to_dict()).T.rename_axis(index='sensitive_feature_0')
conf_matrix.columns  # MultiIndex Columns

# IMPORTANT Given a preds_df, calculate bias scores for given mesaures using Keidar code
# create variables needed for Keidar

sensitive_relations, target_relations = get_sensitive_and_target_relations(args.dataset_name)

# drop columns not needed
preds_df_for_Keidar_code = preds_df

assert preds_df.empty is False
measures = [DemographicParity(), PredictiveParity()]
evaluator = BiasEvaluator(measures)
evaluator.set_predictions_df(preds_df_for_Keidar_code)
bias_eval = evaluator.evaluate_bias(bias_relations = sensitive_relations, bias_measures = measures)
d_parity, p_parity = bias_eval['demographic_parity'], bias_eval['predictive_parity']

from src.bias_measurement.link_prediction_bias.utils import save_result

# to Save dataset summary, and output from Evaluator

# save_result(result = bias_eval, dataset = dataset, args = args)

logger.info('Finished running the script.')
