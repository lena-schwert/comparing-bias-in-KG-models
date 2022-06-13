# This script is based on: https://github.com/mianzg/kgbiasdetec
# More specifically, the following files:
#   experiments/run_tail_prediction.py
#   predict_tails.py

### Imports

# in-built modules
import argparse
import functools
import logging
import os
import shutil
import socket
from datetime import datetime

# imports from installed libraries
import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import MetricFrame, count
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

# imports from my own code
from src.utils import set_base_path_based_on_host, initialize_my_logger, \
    improve_pandas_viewing_options, save_argparse_obj_to_disk

BASE_PATH_HOST = set_base_path_based_on_host()
improve_pandas_viewing_options()

START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")

# %% CODE CHAPTER: adapt selection_rate (for demographic parity) to the multi-class case
# this is based on the Keidar metric
def calculate_selection_rate(preds_df, sensitive_relation):
    """
    Calculate demographic parity for each of the relation tail values
    a.k.a. Group fairness/statistical parity/equal acceptance rate
    A classifier satisfies this definition if subjects in both protected and unprotected groups
    have equal probability of being assigned to the positive predicted class : P(d=1|G=m)=P(d=1|G=f)

    Return:
    demographic_parity_distance: float
    """
    # IMPORTANT: bias is only calculated on the triples that have sensitive attribute information!
    # "NA" entries with sensitive attribute value "-1" are filtered out!
    logger.info(
        f'{len(preds_df[preds_df[sensitive_relation] != "-1"])} people have information on this relation')
    logger.info(
        f'{len(preds_df[preds_df[sensitive_relation] == "-1"])} people DO NOT have information on this relation')

    preds_df_only_valid_triples = preds_df[preds_df[sensitive_relation] != "-1"]

    ### prepare data objects needed for the loop that calculates bias
    # extract list of unique true labels from the preds_df, e.g. [0, 1, 2, 3, 4, 5]
    set_of_true_tail_class_label = list(
        set(preds_df_only_valid_triples.true_tail_class_label.values))
    # # only keep IDs that appear both as true_tail and as pred_tail
    set_of_true_tail_class_label = [trgt for trgt in set_of_true_tail_class_label if
                                     trgt in set(preds_df_only_valid_triples.pred_tail_class_label)]
    sensitive_relation_tails_vector = preds_df_only_valid_triples[sensitive_relation]

    # create list of unique tail values for current bias relation
    # e.g. for gender this corresponds to the IDs for "male", "female"
    sensitive_relation_tails_set = list(set(sensitive_relation_tails_vector))

    # return bias_score = 0 if no person has this attribute
    if len(sensitive_relation_tails_set) == 0:
        return 0

    results_dict = {}
    results_df = pd.DataFrame()

    for sensitive_tail_value in sensitive_relation_tails_set:
        # sum demographic_parity_distance over all attributes a, by calculating the difference
        # P[y=target_val|a=relation_val] - p[y=target_val|a!=relation_val]
        # for all true target values
        support_predicted_per_class = []
        selection_rate_per_class = []

        logger.debug(f'Current sensitive tail value: {sensitive_tail_value}')
        for target_val in set_of_true_tail_class_label:
            logger.debug(f'Current target value : {target_val}')
            # identify all rows where entities are predicted to have current target_value occupation
            # alternative to Keidar, use normal dataframe querying
            only_predicted_current_target = preds_df_only_valid_triples.query(
                f'pred_tail_class_label == {target_val}')
            logger.debug(
                f'{len(only_predicted_current_target)} people have this predicted target value.')

            # from this, select only the facts/rows that have the current sensitive attribute value
            # e.g. From the previous vector, only select the rows where the person is female.
            of_these_sens_attr = only_predicted_current_target[
                only_predicted_current_target[sensitive_relation] == sensitive_tail_value]
            logger.debug(
                f'Of these, {len(of_these_sens_attr)} facts DO have this particular sensitive tail value.')

            support_predicted_per_class.append(len(of_these_sens_attr))

            # then select only the facts/rows that are NOT equal to the current attribute value
            of_these_not_sens_attr = only_predicted_current_target[
                only_predicted_current_target[sensitive_relation] != sensitive_tail_value]
            logger.debug(
                f'And {len(of_these_not_sens_attr)} DO NOT have this particular sensitive tail value.')

            assert len(of_these_sens_attr) + len(of_these_not_sens_attr) == len(
                only_predicted_current_target), 'Sanity check failed.'

            # Which ratio of people with this sensitive attribute + target value out of all people having this target value?
            # in math this is: P[y = target_val | a = relation_val]
            prob_y_given_a = len(of_these_sens_attr) / len(only_predicted_current_target)
            selection_rate_per_class.append(prob_y_given_a)

            results_dict[f'selection_rate_{sensitive_tail_value}'] = selection_rate_per_class
            results_dict[f'support_{sensitive_tail_value}'] = support_predicted_per_class

        results_df.loc[sensitive_tail_value, 'support_predicted_per_class'] = str(
            list(support_predicted_per_class))
        results_df.loc[sensitive_tail_value, 'selection_rate_per_class'] = str(
            list(np.round(selection_rate_per_class, 4)))
        results_df.loc[sensitive_tail_value, 'selection_rate_averaged_macro'] = np.mean(
            selection_rate_per_class)


    results_df.loc['difference', 'support_predicted_per_class'] = str(
        list(abs(np.array(results_dict.get(f'support_{sensitive_relation_tails_set[0]}')) - np.array(results_dict.get(f'support_{sensitive_relation_tails_set[1]}')))))
    results_df.loc['difference', 'selection_rate_per_class'] = str(
        list(np.round(abs(np.array(results_dict.get(f'selection_rate_{sensitive_relation_tails_set[0]}')) - np.array(results_dict.get(f'selection_rate_{sensitive_relation_tails_set[1]}'))), 4)))
    results_df.loc['difference', 'selection_rate_averaged_macro'] = abs(
        results_df.loc[sensitive_relation_tails_set[0], 'selection_rate_averaged_macro'] - results_df.loc[sensitive_relation_tails_set[1], 'selection_rate_averaged_macro'])

    return results_df

# %% CODE CHAPTER: parse arguments and create logging file

parser = argparse.ArgumentParser(
    description = 'Calculate predictive bias using a trained link prediction model')

parser.add_argument('--name', type = str,
                    help = 'Name of run, will determine folder name where results are stored')
parser.add_argument('--preds_df_path', type = str,
                    default = 'results/bias_measurement/link_prediction_bias',
                    help = 'Specifies path to an existing preds_df or folder of preds_df. ')

parser.add_argument('-d', "--debug", action = 'store_true', help = "Add this flag when debugging.")

args = parser.parse_args()

if args.debug:
    EXPERIMENT_NAME = 'DEBUGGING_' + START_TIME + '_' + args.name
    if args.dataset_name == 'Wikidata5M':
        # test has 10 examples, training has
        args.data_dir = os.path.join(args.data_dir, 'for_debugging')
else:
    EXPERIMENT_NAME = START_TIME + '_' + args.name

DIRECTORY_FOR_SAVING_OR_LOADING = os.path.join(BASE_PATH_HOST,
                                               'results/bias_measurement/link_prediction_bias/log_folders',
                                               EXPERIMENT_NAME)

# create directory and then use it as working directory
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
logger = initialize_my_logger(file_name = logger_file_name, level = logging.DEBUG)
logger.info(f'Saving everything in folder: {DIRECTORY_FOR_SAVING_OR_LOADING}')

if args.debug:
    logger.info(f'DEBUGGING MODE: Using a very small subset of {args.dataset_name}.')

logger.info('################# DEVICE INFORMATION #################')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device used: {device}')
logger.info(f'CUDA is used: {torch.cuda.is_available()}')
logger.info(f'Using {torch.cuda.device_count()} device(s).')

LIST_OF_PREDS_DF_PATHS_occupation = [
    'KGE_models_subset_HumanW5M/preds_df_pykeen_TransE_HW5M_subset_occupation_9classes_WITHOUTNAs_sensrel_from_entire_v4subset.tsv',
    'KGE_models_subset_HumanW5M/preds_df_pykeen_DistMult_HW5M_subset_occupation_9classes_WITHOUTNAs_sensrel_from_entire_v4subset.tsv',
    'KGE_models_subset_HumanW5M/preds_df_pykeen_RotatE_HW5M_subset_occupation_9classes_WITHOUTNAs_sensrel_from_entire_v4subset.tsv',
    'SimKGC_IB/preds_df_SimKGC_IB_HW5M_occupation_9classes_WITHOUTNAs_sensrel_from_entire_v4subset.tsv',
    'SimKGC_IBSNPB/preds_df_SimKGC_IBSNPB_HW5M_occupation_9classes_WITHOUTNAs_sensrel_from_entire_v4subset.tsv',
    'KGE_models_FB15k-237/preds_df_Rossi_TransE_FB15k237_occupation_9classes_DOESNOTHAVENAs_sensrel_from_entire_FB15k-237.tsv',
    'KGE_models_FB15k-237/preds_df_Rossi_DistMult_FB15k237_occupation_9classes_DOESNOTHAVENAs_sensrel_from_entire_FB15k-237.tsv',
    'KGE_models_FB15k-237/preds_df_Rossi_RotatE_FB15k237_occupation_9classes_WITHOUTNAs_sensrel_from_entire_FB15k-237.tsv',
    'KG-BERT/preds_df_KG-BERT_FB15k237_occupation_9classes_WITHOUTNAs_sensrel_from_entire_FB15k-237.tsv']

# LIST_OF_PREDS_DF_PATHS_occupation = [
#     'preds_df_Rossi_TransE_FB15k237_occupation_9classes_DOESNOTHAVENAs_sensrel_from_entire_FB15k-237.tsv'
# ]

LIST_OF_PREDS_DF_PATHS = LIST_OF_PREDS_DF_PATHS_occupation

# automatically detect dataset name from the preds_df file path
SENSITIVE_ATTRIBUTE = 'gender'

results_df_for_all_models = pd.DataFrame()

for preds_df_path in LIST_OF_PREDS_DF_PATHS:

    results_df_for_this_model = pd.DataFrame()

    logger.info(f'Calculating link prediction bias for file:{preds_df_path}')

    # automatically detect dataset name from the preds_df file path
    if SENSITIVE_ATTRIBUTE == 'gender':
        if 'HW5M' in preds_df_path:
            SENSITIVE_ATTRIBUTE_COLUMN = 'P21'
            MALE_ENTITY = 'Q6581097'
            FEMALE_ENTITY = 'Q6581072'
        elif 'FB15k237' in preds_df_path:
            SENSITIVE_ATTRIBUTE_COLUMN = '/people/person/gender'
            MALE_ENTITY = '/m/05zppz'
            FEMALE_ENTITY = '/m/02zsn'
        else:
            raise ValueError('Check file name of preds_df path, it must contain the dataset name.')
    else:
        raise NotImplementedError()

    preds_df = pd.read_csv(os.path.join(BASE_PATH_HOST, args.preds_df_path, preds_df_path),
                           sep = '\t')

    # IMPORTANT Given a preds_df, calculate bias scores using fairlearn toolkit

    preds_df_only_valid_triples = preds_df[preds_df[SENSITIVE_ATTRIBUTE_COLUMN] != "-1"]
    if len(preds_df_only_valid_triples[SENSITIVE_ATTRIBUTE_COLUMN].unique()) != 2:
        raise NotImplementedError(
            'Implementation currently works only for binary sensitive attributes!')

    # do the same for precision (= predictive parity)
    for average_method in ['micro', 'macro']:
        multiclass_precision_averaged = functools.partial(precision_score, average = average_method)

        # do the same for recall (= equality of opportunity)
        # equal TPR = equal FNR mathematically, and TPR = recall = sensitivity
        multiclass_recall_averaged = functools.partial(recall_score, average = average_method)

        recall_precision_count = MetricFrame(
            metrics = {f'precision_averaged_{average_method}': multiclass_precision_averaged,
                       f'recall_averaged_{average_method}': multiclass_recall_averaged,
                       f'count': count}, y_true = preds_df_only_valid_triples.true_tail_class_label,
            y_pred = preds_df_only_valid_triples.pred_tail_class_label,
            sensitive_features = preds_df_only_valid_triples[SENSITIVE_ATTRIBUTE_COLUMN])

        results_df_for_this_model = pd.concat([results_df_for_this_model, recall_precision_count.by_group],
                                              axis = 1)
        results_df_for_this_model.loc['difference'] = recall_precision_count.difference()

    # demographic parity: calculate selection rate for each class
    selection_rate_df = calculate_selection_rate(preds_df = preds_df,
                                                 sensitive_relation = SENSITIVE_ATTRIBUTE_COLUMN)

    results_df_for_this_model = pd.concat([results_df_for_this_model, selection_rate_df],
                                          axis = 1)

    # multiclass accuracy disparity: returns accuracy for each sensitive group
    acc = MetricFrame(metrics = accuracy_score,  # sample_params = dict(average = None),
                      y_true = preds_df_only_valid_triples.true_tail_class_label,
                      y_pred = preds_df_only_valid_triples.pred_tail_class_label,
                      sensitive_features = preds_df_only_valid_triples[SENSITIVE_ATTRIBUTE_COLUMN])

    results_df_for_this_model[f'accuracy'] = acc.by_group
    results_df_for_this_model.loc['difference', 'accuracy'] = abs(
        results_df_for_this_model.loc[MALE_ENTITY, 'accuracy'] -
        results_df_for_this_model.loc[FEMALE_ENTITY, 'accuracy'])

    # # IMPORTANT create results that are not averaged across the target property classes

    from sklearn.metrics import precision_recall_fscore_support, classification_report

    only_male_triples = preds_df_only_valid_triples[preds_df_only_valid_triples[SENSITIVE_ATTRIBUTE_COLUMN] == MALE_ENTITY]
    only_female_triples = preds_df_only_valid_triples[preds_df_only_valid_triples[SENSITIVE_ATTRIBUTE_COLUMN] == FEMALE_ENTITY]

    precision_male, recall_male, fscore_male, support_male = precision_recall_fscore_support(
        y_true = only_male_triples.true_tail_class_label,
        y_pred = only_male_triples.pred_tail_class_label,
        average = None, labels = [*range(0, 9, 1)]
    )

    precision_female, recall_female, fscore_female, support_female = precision_recall_fscore_support(
        y_true = only_female_triples.true_tail_class_label,
        y_pred = only_female_triples.pred_tail_class_label,
        average = None, labels = [*range(0, 9, 1)]
    )

    classification_report_male = classification_report(
        y_true = only_male_triples.true_tail_class_label,
        y_pred = only_male_triples.pred_tail_class_label
    )
    print(classification_report_male)

    classification_report_female = classification_report(
        y_true = only_female_triples.true_tail_class_label,
        y_pred = only_female_triples.pred_tail_class_label
    )
    print(classification_report_male)

    results_df_for_this_model.loc[MALE_ENTITY, 'precision_per_class'] = str(list(np.round(precision_male, 4)))
    results_df_for_this_model.loc[FEMALE_ENTITY, 'precision_per_class'] = str(list(np.round(precision_female, 4)))
    results_df_for_this_model.loc['difference', 'precision_per_class'] = str(
        list(np.round(abs(precision_male - precision_female), 4)))

    results_df_for_this_model.loc[MALE_ENTITY, 'recall_per_class'] = str(list(np.round(recall_male, 4)))
    results_df_for_this_model.loc[FEMALE_ENTITY, 'recall_per_class'] = str(list(np.round(recall_female, 4)))
    results_df_for_this_model.loc['difference', 'recall_per_class'] = str(
        list(np.round(abs(recall_male - recall_female), 4)))

    results_df_for_this_model.loc[MALE_ENTITY, 'support_per_class'] = str(list(np.round(support_male, 4)))
    results_df_for_this_model.loc[FEMALE_ENTITY, 'support_per_class'] = str(list(np.round(support_female, 4)))
    results_df_for_this_model.loc['difference', 'support_per_class'] = str(
        list(np.round(abs(support_male - support_female), 4)))

    # IMPORTANT save all results in the respective working directory, not in the folders where I load the files from
    # store results in dataframe and write to TSV file
    results_df_for_this_model['model_class'] = os.path.dirname(preds_df_path)
    results_df_for_this_model['file_name'] = os.path.basename(preds_df_path)

    results_df_for_all_models = pd.concat([results_df_for_all_models, results_df_for_this_model])

# keep header and index! They store important information about the sensitive attribute
results_df_for_all_models.to_csv(f'results_from_all_preds_df_{START_TIME}.tsv', sep = '\t')


# try a confusion matrix
# source: https://github.com/fairlearn/fairlearn/issues/752

# pandified confusion matrix with clarifying column annotations
def confusion_matrix_pd(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, normalize = 'all')
    return pd.DataFrame(matrix).rename(
        index = {0: "true=0", 1: "true=1", 2: "true=2", 3: "true=3", 4: "true=4", 5: "true=5",
                 6: "true=6", 7: "true=7", 8: "true=8"},
        columns = {0: "pred=0", 1: "pred=1", 2: "pred=2", 3: "pred=3", 4: "pred=4", 5: "pred=5",
                   6: "pred=6", 7: "pred=7", 8: "pred=8"}).stack()


# construct metric frame
cf_mtrx = MetricFrame(metrics = confusion_matrix_pd,
                 y_true = preds_df_only_valid_triples.true_tail_class_label,
                 y_pred = preds_df_only_valid_triples.pred_tail_class_label,
                 sensitive_features = preds_df_only_valid_triples[SENSITIVE_ATTRIBUTE_COLUMN])

# mf.by_group needs to be transformed to work properly
conf_matrix = pd.DataFrame(cf_mtrx.by_group.to_dict()).T.rename_axis(index = 'sensitive_feature_0')
conf_matrix.columns  # MultiIndex Columns

logger.info('Finished running the script.')
