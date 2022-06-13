# This script is based on: https://github.com/merialdo/research.lpbias
# Scripts are used with permission of Andrea Rossi.
# Paper: Rossi, A., Firmani, D. and Merialdo, P. (2021) ‘Knowledge Graph Embeddings or Bias
# Graph Embeddings? A Study of Bias in Link Prediction Models’, in DL4KG 2021:
# Workshop on Deep Learning for Knowledge Graphs, held as part of ISWC 2021: The 20th
# International Semantic Web Conference.

### Imports

# in-built modules
import logging
import os
import shutil
import socket
import sys
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# Installed packages
import pandas as pd
import numpy as np

# Internal imports
from src.utils import set_base_path_based_on_host, initialize_my_logger, \
    improve_pandas_viewing_options
from bias_measurement.data_bias.Rossi_bias_types.dataset import Dataset, MANY_TO_ONE, ONE_TO_MANY, MANY_TO_MANY
from bias_measurement.data_bias.Rossi_bias_types.config import SELECTED_DATASET_NAMES, BIAS_DATA_PATH

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

BASE_PATH_HOST = set_base_path_based_on_host()
improve_pandas_viewing_options()

START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")
EXPERIMENT_NAME = START_TIME


DIRECTORY_FOR_SAVING_OR_LOADING = os.path.join(BASE_PATH_HOST, BIAS_DATA_PATH, EXPERIMENT_NAME)

DEBUGGING = True

os.makedirs(DIRECTORY_FOR_SAVING_OR_LOADING)
os.chdir(DIRECTORY_FOR_SAVING_OR_LOADING)

# save the currently running script file for later reference
file_name_script = 'script_get_bias_by_type' + EXPERIMENT_NAME + '.py'
source_path = os.path.join(BASE_PATH_HOST, 'src/CLI_scripts', __file__)
shutil.copy(src = source_path, dst = os.path.join(os.getcwd(), file_name_script))

# initialize logger
logger_file_name = f'log_get_bias_by_type_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'
logger = initialize_my_logger(file_name = logger_file_name, level = logging.DEBUG)
logger.info(f'Saving everything in folder: {DIRECTORY_FOR_SAVING_OR_LOADING}')

# TODO possible CLI options
# threshold
# data split paths


def extract_all_3_bias_types_from_dataset(thresholds_iterable):
    for dataset_name in SELECTED_DATASET_NAMES:
        for THRESHOLD in thresholds_iterable:
            logger.info(f'Calculating bias for threshold {THRESHOLD}...')

            dataset = Dataset(name = dataset_name, debugging = DEBUGGING)

            result_list_bias_type_1 = extract_bias_type_1(dataset, dataset_name, THRESHOLD)

            result_list_bias_type_2 = extract_bias_type_2(dataset, dataset_name, THRESHOLD)

            result_list_bias_type_3 = extract_bias_type_3(dataset, dataset_name, THRESHOLD)

            results_per_threshold = result_list_bias_type_1 + result_list_bias_type_2 + result_list_bias_type_3

            # save one file per threshold to be used in summarizing results
            results_per_threshold_df = pd.DataFrame(results_per_threshold,
                                                    columns = ['threshold', 'head', 'relation',
                                                               'tail', 'is_head_biased', 'is_tail_biased'])
            # save to csv with START_TIME in name
            results_per_threshold_df.to_csv(os.path.join(START_TIME,  f'all_bias_types_thresh_{THRESHOLD}.csv'),
                                            header = False, index = False)

            # Create global variable name per threshold and dataset
            dynamic_variable_name = f'{dataset_name}_results_thresh_{THRESHOLD}'
            globals()[dynamic_variable_name] = results_per_threshold_df


def extract_bias_type_1(dataset, dataset_name, threshold):
    # equivalent to scripts/extract_biased_test_facts_type_1.py

    logger.info("Identifying test predictions prone to bias type 1 in dataset " + dataset_name + "...")

    outlines = []

    train_triples = dataset.train_triples
    test_triples = dataset.test_triples

    # map each relation to the number of times that relation is used in the training set
    relation_2_count = defaultdict(lambda: 0)

    # map each relation to the number of times any entity is used as head in train facts with that relation
    relation_2_head_counts = defaultdict(lambda: defaultdict(lambda: 0))

    # map each relation to the number of times any entity is used as tail in train facts with that relation
    relation_2_tail_counts = defaultdict(lambda: defaultdict(lambda: 0))

    for (h, r, t) in train_triples:
        relation_2_head_counts[r][h] += 1
        relation_2_tail_counts[r][t] += 1
        relation_2_count[r] += 1

    def is_a_biased_tail_prediction(head, relation, tail):
        return float(relation_2_tail_counts[relation][tail]) / float(relation_2_count[relation]) >= threshold

    def is_a_biased_head_prediction(head, relation, tail):
        return float(relation_2_head_counts[relation][head]) / float(relation_2_count[relation]) >= threshold

    for (x, y, z) in tqdm(test_triples):
        biased_head_prediction = is_a_biased_head_prediction(x, y, z)
        biased_tail_prediction = is_a_biased_tail_prediction(x, y, z)

        biased_head_prediction_str = "0"
        if biased_head_prediction:
            biased_head_prediction_str = "1"

        biased_tail_prediction_str = "0"
        if biased_tail_prediction:
            biased_tail_prediction_str = "1"

        outlines.append(",".join([str(threshold), x, y, z, biased_head_prediction_str, biased_tail_prediction_str]) + "\n")

    output_path = os.path.join(BIAS_DATA_PATH, dataset_name + f"_test_set_b1_thresh_{threshold}.csv")
    with open(output_path, "w") as outfile:
        outfile.writelines(outlines)

    return outlines


def extract_bias_type_2(dataset, dataset_name, threshold):
    # equivalent to scripts/extract_biased_test_facts_type_2.py

    logger.info("Identifying test predictions prone to bias type 2 in dataset " + dataset_name + "...")

    outlines = []

    train_triples = dataset.train_triples
    test_triples = dataset.test_triples

    # map each relation to the number of times that relation is used in the training set
    relation_2_count = defaultdict(lambda: 0)

    # map each relation to the number of times any entity is used as head in train facts with that relation
    relation_2_head_counts = defaultdict(lambda: defaultdict(lambda: 0))

    # map each relation to the number of times any entity is used as tail in train facts with that relation
    relation_2_tail_counts = defaultdict(lambda: defaultdict(lambda: 0))

    relation_2_heads = defaultdict(lambda: set())
    relation_2_tails = defaultdict(lambda: set())
    relation_and_head_2_tails = defaultdict(lambda: set())
    relation_and_tail_2_heads = defaultdict(lambda: set())

    for (h, r, t) in train_triples:
        relation_2_heads[r].add(h)
        relation_2_tails[r].add(t)
        relation_and_head_2_tails[(r, h)].add(t)
        relation_and_tail_2_heads[(r, t)].add(h)
        relation_2_count[r] += 1

    def is_a_biased_tail_prediction(head, relation, tail):
        if dataset.relation_2_type[relation] in [ONE_TO_MANY, MANY_TO_MANY]:

            #let's say the passed fact is <Barack Obama, speaks_language, English>

            # select the entities that are seen as heads in facts with the relation
            # e.g. <A, speaks_language, any_language>
            heads_involved_with_relation = relation_2_heads[relation]

            # select the entities that are seen as heads in facts <_, relation, tail>
            # e.g. <A, speaks_language, English>
            heads_involved_with_relation_and_tail = relation_and_tail_2_heads[(relation, tail)]

            assert len(heads_involved_with_relation_and_tail) <= len(heads_involved_with_relation)
            # the idea is that if most A entities that speak any language speak English too, then there is a bias
            return float(len(heads_involved_with_relation_and_tail)) / float(len(heads_involved_with_relation)) >= threshold
        return False

    def is_a_biased_head_prediction(head, relation, tail):
        if dataset.relation_2_type[relation] in [MANY_TO_ONE, MANY_TO_MANY]:
            #let's say the passed fact is <USA, contain, Washington>

            # select the entities that are seen as tails in facts with the relation
            # e.g. <any_location, contains, A>
            tails_involved_with_relation = relation_2_tails[relation]

            # select the entities that are seen as tails in facts <head, relation, _>
            # e.g. <Washington, contain, A>
            tails_involved_with_relation_and_head = relation_and_head_2_tails[(relation, head)]

            # the idea is that if most A entities that are contained in a location are contained in USA too, there is a bias
            return float(len(tails_involved_with_relation_and_head)) / float(len(tails_involved_with_relation)) >= threshold
        return False

    for (x, y, z) in tqdm(test_triples):
        biased_head_prediction = is_a_biased_head_prediction(x, y, z)
        biased_tail_prediction = is_a_biased_tail_prediction(x, y, z)

        biased_head_prediction_str = "0"
        if biased_head_prediction:
            biased_head_prediction_str = "1"

        biased_tail_prediction_str = "0"
        if biased_tail_prediction:
            biased_tail_prediction_str = "1"

        outlines.append(",".join([str(threshold), x, y, z, biased_head_prediction_str, biased_tail_prediction_str]) + "\n")

    output_path = os.path.join(BIAS_DATA_PATH, dataset_name + f"_test_set_b2_thresh_{threshold}.csv")
    with open(output_path, "w") as outfile:
        outfile.writelines(outlines)

    return outlines


def extract_bias_type_3(dataset, dataset_name, threshold):
    # equivalent to scripts/extract_biased_test_facts_type_3.py

    outlines = []

    logger.info("Identifying test predictions prone to bias type 3 in dataset " + dataset_name + "...")

    train_triples = dataset.train_triples
    test_triples = dataset.test_triples

    # map each relation to the number of times that relation is used in the training set
    relation_2_count = defaultdict(lambda: 0)

    # map each head and tail to the relations connecting them
    head_and_tail_2_relations = defaultdict(lambda: set())

    # map each relation to the heads and tails that it connects
    relation_to_heads_and_tails = defaultdict(lambda: set())
    for (h, r, t) in train_triples:
        head_and_tail_2_relations[(h, t)].add(r)
        relation_to_heads_and_tails[r].add((h, t))
        relation_2_count[r] += 1

    relation_2_dominating_relations = defaultdict(lambda: set())
    for r1 in dataset.relations:
        r1_heads_and_tails = relation_to_heads_and_tails[r1]
        for r2 in dataset.relations:
            r1_r2_count = 0

            if r2 == r1:
                continue

            r2_heads_and_tails = relation_to_heads_and_tails[r2]

            for r1_head_and_tail in r1_heads_and_tails:
                if r1_head_and_tail in r2_heads_and_tails:
                    r1_r2_count += 1

            if float(r1_r2_count) / float(len(r2_heads_and_tails)) > threshold:
                relation_2_dominating_relations[r1].add(r2)

    def is_a_biased_tail_prediction(head, relation, tail):

        dominating_relations = relation_2_dominating_relations[relation]

        for dominating_relation in dominating_relations:
            if (head, dominating_relation, tail) in dataset.train_triples_set:
                return True
        return False

    def is_a_biased_head_prediction(head, relation, tail):
        dominating_relations = relation_2_dominating_relations[relation]

        for dominating_relation in dominating_relations:
            if (head, dominating_relation, tail) in dataset.train_triples_set:
                return True
        return False

    for (x, y, z) in tqdm(test_triples):
        biased_head_prediction = is_a_biased_head_prediction(x, y, z)
        biased_tail_prediction = is_a_biased_tail_prediction(x, y, z)

        biased_head_prediction_str = "0"
        if biased_head_prediction:
            biased_head_prediction_str = "1"

        biased_tail_prediction_str = "0"
        if biased_tail_prediction:
            biased_tail_prediction_str = "1"

        outlines.append(",".join([str(threshold), x, y, z, biased_head_prediction_str, biased_tail_prediction_str]) + "\n")

    output_path = os.path.join(BIAS_DATA_PATH, dataset_name + f"_test_set_b3_thresh_{threshold}.csv")
    with open(output_path, "w") as outfile:
        outfile.writelines(outlines)

    return outlines


############################################

long_list_of_thresholds = np.arange(0.05, 1.05, 0.05)
short_list_of_thresholds = np.arange(0.05, 1.05, 0.1)
# 20 thresholds for steps of 0.05 np.arange(0.05, 1.05, 0.05)

calculate_from_scratch = True

if calculate_from_scratch:
    extract_all_3_bias_types_from_dataset(short_list_of_thresholds)
    # after running this, one dataframe variable exists per threshold
    # naming pattern: f'{dataset_name}_results_thresh_{THRESHOLD}'

if not calculate_from_scratch:
    raise NotImplementedError
    # Load the respective counts for each dataset: use files in "scripts" folder to get them
    # bias_type_1_raw = pd.read_csv(os.path.join(BASE_PATH_HOST,
    #                                            'results/bias_measurement/data_bias/Rossi_bias_types/HumanWikidata5M_test_set_b1.csv'),
    #                               sep = ';',
    #                               names = ['head', 'relation', 'tail', 'is_head_biased',
    #                                        'is_tail_biased'])
    # bias_type_2_raw = pd.read_csv(os.path.join(BASE_PATH_HOST,
    #                                            'results/bias_measurement/data_bias/Rossi_bias_types/HumanWikidata5M_test_set_b2.csv'),
    #                               sep = ',',
    #                               names = ['head', 'relation', 'tail', 'is_head_biased',
    #                                        'is_tail_biased'])
    # bias_type_3_raw = pd.read_csv(os.path.join(BASE_PATH_HOST,
    #                                            'results/bias_measurement/data_bias/Rossi_bias_types/HumanWikidata5M_test_set_b3.csv'),
    #                               sep = ',',
    #                               names = ['head', 'relation', 'tail', 'is_head_biased',
    #                                        'is_tail_biased'])

    # check size with wc-l, train: 8891836, valid: 1111480, test: 1111480
    total_number_of_test_triples = len(bias_type_1_raw)
    assert total_number_of_test_triples == 1111480

# calculate counts and percentages of how many triples have bias type 1,2,3 for a given threshold

results_summarized = pd.DataFrame(columns = ['bias_type', 'threshold', 'count_biased_both',
                                             'percentage_biased_both', 'count_biased_head', 'count_biased_tail',
                                             'percentage_biased_head', 'percentage_biased_tail'])

results_summarized['bias_type'] = [1, 2, 3]

for (i, dataframe) in enumerate([bias_type_1_raw, bias_type_2_raw, bias_type_3_raw]):

    sum_heads = sum(dataframe['is_head_biased'])
    sum_tails = sum(dataframe['is_tail_biased'])
    results_summarized.loc[i, 'count_biased_head'] = sum_heads
    results_summarized.loc[i, 'count_biased_tail'] = sum_tails
    results_summarized.loc[i, 'percentage_biased_head'] = round(sum_heads/total_number_of_test_triples, 5)
    results_summarized.loc[i, 'percentage_biased_tail'] = round(sum_tails/total_number_of_test_triples, 5)

    results_summarized.loc[i, 'count_biased_both'] = sum_heads + sum_tails
    results_summarized.loc[i, 'percentage_biased_both'] = round((sum_heads + sum_tails)/(2 * total_number_of_test_triples), 5)


results_summarized['threshold'] = 0.75


results_summarized.to_csv(os.path.join(BASE_PATH_HOST, 'results/bias_measurement/data_bias/Rossi_bias_types/threshold_0.75_summarized.csv'))
