# %% Imports
# In-built modules
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import pykeen
import torch

# Internal Imports
from src.utils import set_base_path_based_on_host, HumanWikidata5M_pykeen

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 750)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 20)

BASE_PATH_HOST = set_base_path_based_on_host()

# %% Set target and sensitive relation(s)

# SENSITIVE_RELATIONS, TARGET_RELATIONS = get_sensitive_and_target_relations(
#     dataset_name = 'Wikidata5m')

COLUMN_NAMES_PREDS_DF = ['head_id', 'relation_id', 'relation_label', 'true_tail_id',
                         'true_tail_label', 'pred_tail_id', 'pred_tail_label']


# %% function for extracting sensitive attribute information from a given pykeen dataset


def add_sensitive_relation_values(dataset, preds_df, sensitive_relations):
    """
    CREDIT: This function is a slightly adapted version from here: https://github.com/mianzg/kgbiasdetec/blob/main/predict_tails.py
    associated paper: Keidar et al. (2021): Towards Automatic Bias Detection in Knowledge Graphs. EMNLP

    Given a dataframe with predictions for the target relation, add the tail values
    for each sensitive relation for each human head entity in the preds_df.
    Adds one column per sensitive relation.
    Value is either the numeric ID of the tail entity or -1 if the human does not have
    a fact about this specific sensitive relation.

    dataset: subclass of pykeen.Dataset, knowledge graph dataset e.g HumanWikidata5M_pykeen, FB15k237
    preds_df: pd.DataFrame,
    bias_relations: list of str
    """

    def get_tail(rel, x):
        """

        Parameters
        ----------
        rel
        x

        Returns
        -------
        The num ID of the corresponding tail entity, or -1 if this
        """
        try:
            return entity_to_tail[rel][x]
        except KeyError:
            return -1

    assert type(sensitive_relations) == list
    assert isinstance(dataset, pykeen.datasets.Dataset)

    # TODO change this, access the triples from all dataset splits! Create a union of all splits

    # access the test triples of the dataset which have the bias relations (as string ID triples)
    bias_relations_triplets_mask_training = dataset.training.get_mask_for_relations(
        sensitive_relations)
    bias_relations_triplets_training = dataset.training.triples[
        bias_relations_triplets_mask_training]
    bias_relations_triplets_mask_validation = dataset.validation.get_mask_for_relations(
        sensitive_relations)
    bias_relations_triplets_validation = dataset.validation.triples[
        bias_relations_triplets_mask_validation]
    bias_relations_triplets_mask_testing = dataset.testing.get_mask_for_relations(
        sensitive_relations)
    bias_relations_triplets_testing = dataset.testing.triples[bias_relations_triplets_mask_testing]

    uniques, counts = np.unique(bias_relations_triplets_testing[:, 1], return_counts = True)

    bias_relations_triplets_entire_dataset = np.concatenate((bias_relations_triplets_training,
                                                             bias_relations_triplets_validation,
                                                             bias_relations_triplets_testing),
                                                            axis = 0)

    # only select the bias_relation facts for which the human head entity is part of preds_df['entity']
    # bias_relations_triplets = [tr for tr in bias_relations_triplets if dataset.entity_to_id[tr[0]] in preds_df['head_entity'].values]
    entity_to_tail = {}
    # for each bias relation, create a key-value pair, where:
    # key = bias relation string, value = empty dict
    # e.g. {'P21': {}}
    for bias_rel in sensitive_relations:
        entity_to_tail[bias_rel] = {}
    # for each bias relation triple, add the numeric head and tail ID to entity_to_tail
    # e.g. {'P21': {379209: 1370033, 763948: 1370033}}
    for head, rel, tail in bias_relations_triplets_entire_dataset:
        # retrieve numeric ID for head and tail entity
        # head_id = dataset.entity_to_id.get(head)
        # tail_id = dataset.entity_to_id.get(tail)
        # create a dict entry, where key = num ID head, value = num ID tail
        # entity_to_tail[rel][head_id] = tail_id
        # new code
        entity_to_tail[rel][head] = tail
    # for each bias relation, create a column of numeric ID tail values for the corresponding human head entity
    for bias_rel in sensitive_relations:
        # for each head_entity in preds_df, retrieve the numeric ID for the corresponding tail
        # value will be -1 if this fact does not exist in the test set
        # TODO maybe retrieve the sensitive relation facts from the entire dataset instead?
        preds_df[bias_rel] = [get_tail(bias_rel, head_entity) for head_entity in
                              preds_df['head_id'].values]
        # count the occurrence
        attr_counts = Counter(preds_df[bias_rel])
        print(attr_counts)
    return preds_df

# %% CODE CHAPTER: Process pykeen evaluation results (KG only + HumanWikidata5M)

path_to_results = os.path.join(BASE_PATH_HOST, 'results/KG_only/final')
experiment_name = '23.05.2022_20:05_final_model_TransE_512dim_32ns_1024bs_0.001lr_after_400ep'
file_name = 'trained_model.pkl'
# 23.05.2022_20:05_final_model_TransE_512dim_32ns_1024bs_0.001lr_after_400ep
# 23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_after_400ep
# 23.05.2022_20:06_final_model_RotatE_512dim_32ns_1024bs_0.01lr_after_400ep

### IMPORTANT: For each model, get the tail predictions for the target relation triples in the testset
trained_model = torch.load(os.path.join(path_to_results, experiment_name, file_name),
                           map_location = 'cpu')
# retrieve the training Triples
rel_training_set_path = 'data/processed/output_of_preprocessing/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv'
rel_validation_set_path = 'data/processed/output_of_preprocessing/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
rel_test_set_path = 'data/processed/output_of_preprocessing/test_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
dataset_HumanWikidata5M = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                                 rel_training_set_path = rel_training_set_path,
                                                 rel_validation_set_path = rel_validation_set_path,
                                                 rel_test_set_path = rel_test_set_path)

# create numpy array of label-based triples in the testset
# template file with the 3763 occupation triples: preds_df_occupation_HumanW5M_subset_testset.tsv
occupation_triples = pd.read_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'preds_df_occupation_HumanW5M_subset_testset.tsv'), sep = '\t',
    usecols = [0, 1, 3])
# occupation_triples_array = occupation_triples.to_numpy(dtype = str)

# alternatively, try to use functions usually used with PipelineResult:
# https://pykeen.readthedocs.io/en/stable/api/pykeen.models.predict.predict_triples_df.html
from pykeen.models.predict import get_tail_prediction_df

predicted_tail_entity_IDs = []
predicted_tail_entity_labels = []

# IMPORTANT: careful, running this takes about 2-4 hours per model on a A100 GPU!
for rowtuple in occupation_triples.itertuples():
    #    print(rowtuple)
    prediction_result = get_tail_prediction_df(model = trained_model, head_label = rowtuple.head_id,
                                               relation_label = rowtuple.relation_id,
                                               triples_factory = dataset_HumanWikidata5M.training,
                                               testing = dataset_HumanWikidata5M.testing.mapped_triples,
                                               remove_known = True)
    highest_score_row = prediction_result.iloc[:1]
    predicted_tail_entity_IDs.append(highest_score_row['tail_label'].item())
    predicted_tail_entity_labels.append(
        dataset_HumanWikidata5M.entity_numID_to_label.get(highest_score_row['tail_id'].item()))

print('Finished extracting the tail entity predictions.')

pykeen_HW5M_results_for_bias_measurement = occupation_triples.copy()

pykeen_HW5M_results_for_bias_measurement['pred_tail_id'] = predicted_tail_entity_IDs
pykeen_HW5M_results_for_bias_measurement['pred_tail_label'] = predicted_tail_entity_labels

# pykeen_HW5M_results_for_bias_measurement.to_csv(
#     os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
#                  'predsdf_pykeen_TransE_top1predictions_occupation.tsv'),
#     sep = '\t', index = False)

### IMPORTANT: recode the tail values into a fixed set of classes
file_for_class_encoding = 'src/bias_measurement/link_prediction_bias/' \
                          'target_relation_encodings/occupation_P106_9classes_using_v4testset.tsv'
target_relation_class_encoding = pd.read_csv(os.path.join(BASE_PATH_HOST, file_for_class_encoding),
                                             sep = '\t', usecols = [0, 1, 2, 3, 4])
target_relation_class_encoding = target_relation_class_encoding.convert_dtypes()

experiment_name = '23.05.2022_20:06_final_model_RotatE_512dim_32ns_1024bs_0.01lr_after_400ep'
# 23.05.2022_20:05_final_model_TransE_512dim_32ns_1024bs_0.001lr_after_400ep
# 23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_after_400ep
# 23.05.2022_20:06_final_model_RotatE_512dim_32ns_1024bs_0.01lr_after_400ep
file_name = 'pykeen_RotatE_top1predictions_occupation_31052022.tsv'
# pykeen_TransE_top1predictions_occupation_31052022.tsv
# pykeen_DistMult_top1predictions_occupation_31052022.tsv
# pykeen_RotatE_top1predictions_occupation_31052022.tsv

pykeen_HW5M_results_for_bias_measurement = pd.read_csv(
    os.path.join(path_to_results, experiment_name, file_name), sep = '\t')

# retrieve the numeric class labels for 'true_tail_id' and 'pred_tail_id' using merge()
pykeen_HW5M_results_for_bias_measurement = pd.merge(left = pykeen_HW5M_results_for_bias_measurement,
                                                    right = target_relation_class_encoding[
                                                        ['tail_entity_id',
                                                         'class_label_based_on_v4testset']],
                                                    how = 'left', left_on = 'true_tail_id',
                                                    right_on = 'tail_entity_id')
pykeen_HW5M_results_for_bias_measurement.rename(
    columns = {'class_label_based_on_v4testset': 'true_tail_class_label'}, inplace = True)
pykeen_HW5M_results_for_bias_measurement.drop('tail_entity_id', axis = 1, inplace = True)
pykeen_HW5M_results_for_bias_measurement = pd.merge(left = pykeen_HW5M_results_for_bias_measurement,
                                                    right = target_relation_class_encoding[
                                                        ['tail_entity_id',
                                                         'class_label_based_on_v4testset']],
                                                    how = 'left', left_on = 'pred_tail_id',
                                                    right_on = 'tail_entity_id')
pykeen_HW5M_results_for_bias_measurement.rename(
    columns = {'class_label_based_on_v4testset': 'pred_tail_class_label'}, inplace = True)
pykeen_HW5M_results_for_bias_measurement.drop('tail_entity_id', axis = 1, inplace = True)

# detect dtypes
pykeen_HW5M_results_for_bias_measurement = pykeen_HW5M_results_for_bias_measurement.convert_dtypes()
# reorder columns
pykeen_HW5M_results_for_bias_measurement = pykeen_HW5M_results_for_bias_measurement[
    ['head_id', 'relation_id', 'relation_label', 'true_tail_id', 'true_tail_label',
     'true_tail_class_label', 'pred_tail_id', 'pred_tail_label', 'pred_tail_class_label']]

# IMPORTANT count and exclude rows where predicted entity is not an occupation tail entity!
# this contains for TransE 28 rows, DistMult 5 rows, RotatE 54 rows
NA_rows_pykeen_HW5M_pred_tail_class_label = pykeen_HW5M_results_for_bias_measurement[
    pykeen_HW5M_results_for_bias_measurement['pred_tail_class_label'].isnull()]
# this is empty (as expected)
NA_rows_pykeen_HW5M_true_tail_class_label = pykeen_HW5M_results_for_bias_measurement[
    pykeen_HW5M_results_for_bias_measurement['true_tail_class_label'].isnull()]

# dataframe including all 3763 occupation facts
pykeen_HW5M_results_for_bias_measurement_withNAs = pykeen_HW5M_results_for_bias_measurement.copy()
# dataframe filtered for NAs with 3735 rows (TransE), 3758 rows (DistMult), 3709 (RotatE)
pykeen_HW5M_results_for_bias_measurement_withoutNAs = pykeen_HW5M_results_for_bias_measurement.drop(
    index = NA_rows_pykeen_HW5M_pred_tail_class_label.index)

### IMPORTANT: add sensitive attribute information as columns
SENSITIVE_RELATIONS = ['P21', 'P27', 'P172', 'P140']
# gender, country of citizenship, ethnic group, religion
rel_training_set_path = 'data/processed/output_of_preprocessing/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv'
rel_validation_set_path = 'data/processed/output_of_preprocessing/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
rel_test_set_path = 'data/processed/output_of_preprocessing/test_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
dataset = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                 rel_training_set_path = rel_training_set_path,
                                 rel_validation_set_path = rel_validation_set_path,
                                 rel_test_set_path = rel_test_set_path)

pykeen_HW5M_results_for_bias_measurement_withNAs = add_sensitive_relation_values(
    dataset = dataset, preds_df = pykeen_HW5M_results_for_bias_measurement_withNAs,
    sensitive_relations = SENSITIVE_RELATIONS)

pykeen_HW5M_results_for_bias_measurement_withoutNAs = add_sensitive_relation_values(
    dataset = dataset, preds_df = pykeen_HW5M_results_for_bias_measurement_withoutNAs,
    sensitive_relations = SENSITIVE_RELATIONS)

pykeen_HW5M_results_for_bias_measurement_withNAs.to_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'preds_df_pykeen_RotatE_HW5M_subset_occupation_9classes_WITHNAs_sensrel_from_entire_v4subset.tsv'),
    sep = '\t', index = False)

pykeen_HW5M_results_for_bias_measurement_withoutNAs.to_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'preds_df_pykeen_RotatE_HW5M_subset_occupation_9classes_WITHOUTNAs_sensrel_from_entire_v4subset.tsv'),
    sep = '\t', index = False)


# %% Code chapter: Process SimKGC evaluation results (KG+LM + HumanWikidata5M)

# There are two result files that I want to process:
# 12.05.2022_19:34_wiki5m_trans_train_SimKGC_IB_4xa6k5/eval_test_data_subset_0.05.tsv.json_forward_model_last.mdl.json
# 14.05.2022_00:21_wiki5m_trans_train_SimKGC_IBSNPB_4xa6k5/eval_test_data_subset_0.05.tsv.json_forward_model_best.mdl.json

path_to_results = os.path.join(BASE_PATH_HOST, 'results/KG_and_LM/SimKGC/')
experiment_name = '12.05.2022_19:34_wiki5m_trans_train_SimKGC_IB_4xa6k5'
file_name = 'eval_test_data_subset_0.05.tsv.json_forward_model_last.mdl.json'

raw_file_SimKGC = pd.read_json(os.path.join(path_to_results, experiment_name, file_name))

# count for target 'occupation' relation should be as expected
assert raw_file_SimKGC['relation_id'].value_counts()['P106'] == 3763
# count for target 'educated at' relation should be as expected
assert raw_file_SimKGC['relation_id'].value_counts()['P69'] == 8684

### IMPORTANT: select only the target relation triples
SimKGC_results_filtered_for_target_mask = raw_file_SimKGC['relation_id'].isin(TARGET_RELATIONS)
SimKGC_results_filtered_for_target = raw_file_SimKGC[SimKGC_results_filtered_for_target_mask]
# reorder columns
SimKGC_results_filtered_for_target = SimKGC_results_filtered_for_target[
    ['head_id', 'head', 'relation_id', 'relation', 'tail_id', 'tail', 'pred_tail_id', 'pred_tail',
     'topk_scores_labels', 'correct', 'pred_score', 'topk_scores', 'topk_scores_ids', 'rank']]

### IMPORTANT: recode the tail values into a fixed set of classes
file_for_class_encoding = 'src/bias_measurement/link_prediction_bias/' \
                          'target_relation_encodings/occupation_P106_9classes_using_v4testset.tsv'
target_relation_class_encoding = pd.read_csv(os.path.join(BASE_PATH_HOST, file_for_class_encoding),
                                             sep = '\t', usecols = [0, 1, 2, 3, 4])
target_relation_class_encoding = target_relation_class_encoding.convert_dtypes()

SimKGC_results_for_bias_measurement = SimKGC_results_filtered_for_target[
    ['head_id', 'relation_id', 'relation', 'tail_id', 'tail', 'pred_tail_id', 'pred_tail']]
SimKGC_results_for_bias_measurement.columns = COLUMN_NAMES_PREDS_DF

# retrieve the numeric class labels for 'true_tail_id' and 'pred_tail_id' using merge()
SimKGC_results_for_bias_measurement = pd.merge(left = SimKGC_results_for_bias_measurement,
                                               right = target_relation_class_encoding[
                                                   ['tail_entity_id',
                                                    'class_label_based_on_v4testset']],
                                               how = 'left', left_on = 'true_tail_id',
                                               right_on = 'tail_entity_id')
SimKGC_results_for_bias_measurement.rename(
    columns = {'class_label_based_on_v4testset': 'true_tail_class_label'}, inplace = True)
SimKGC_results_for_bias_measurement.drop('tail_entity_id', axis = 1, inplace = True)
SimKGC_results_for_bias_measurement = pd.merge(left = SimKGC_results_for_bias_measurement,
                                               right = target_relation_class_encoding[
                                                   ['tail_entity_id',
                                                    'class_label_based_on_v4testset']],
                                               how = 'left', left_on = 'pred_tail_id',
                                               right_on = 'tail_entity_id')
SimKGC_results_for_bias_measurement.rename(
    columns = {'class_label_based_on_v4testset': 'pred_tail_class_label'}, inplace = True)
SimKGC_results_for_bias_measurement.drop('tail_entity_id', axis = 1, inplace = True)

# detect dtypes
SimKGC_results_for_bias_measurement = SimKGC_results_for_bias_measurement.convert_dtypes()
# reorder columns
SimKGC_results_for_bias_measurement = SimKGC_results_for_bias_measurement[
    ['head_id', 'relation_id', 'relation_label', 'true_tail_id', 'true_tail_label',
     'true_tail_class_label', 'pred_tail_id', 'pred_tail_label', 'pred_tail_class_label']]

# IMPORTANT count and exclude rows where predicted entity is not an occupation tail entity!
# this contains 40 rows (same counts for SimKGC IB and SimKGC IBSNPB)
NA_rows_SimKGC_pred_tail_class_label = SimKGC_results_for_bias_measurement[
    SimKGC_results_for_bias_measurement['pred_tail_class_label'].isnull()]
# this is empty (as expected)
NA_rows_SimKGC_true_tail_class_label = SimKGC_results_for_bias_measurement[
    SimKGC_results_for_bias_measurement['true_tail_class_label'].isnull()]

# dataframe including all 3763 occupation facts (same counts for SimKGC IB and SimKGC IBSNPB)
SimKGC_results_for_bias_measurement_withNAs = SimKGC_results_for_bias_measurement.copy()
# dataframe filtered for NAs with 3723 rows (same counts for SimKGC IB and SimKGC IBSNPB)
SimKGC_results_for_bias_measurement_withoutNAs = SimKGC_results_for_bias_measurement.drop(
    index = NA_rows_SimKGC_pred_tail_class_label.index)

### IMPORTANT: add sensitive attribute information as columns

SENSITIVE_RELATIONS = ['P21', 'P27', 'P172', 'P140']
# gender, country of citizenship, ethnic group, religion
rel_training_set_path = 'data/processed/output_of_preprocessing/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv'
rel_validation_set_path = 'data/processed/output_of_preprocessing/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
rel_test_set_path = 'data/processed/output_of_preprocessing/test_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
dataset = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                 rel_training_set_path = rel_training_set_path,
                                 rel_validation_set_path = rel_validation_set_path,
                                 rel_test_set_path = rel_test_set_path)

preds_df_SimKGC_results_for_bias_measurement_withNAs = add_sensitive_relation_values(
    dataset = dataset, preds_df = SimKGC_results_for_bias_measurement_withNAs,
    sensitive_relations = SENSITIVE_RELATIONS)

preds_df_SimKGC_results_for_bias_measurement_withoutNAs = add_sensitive_relation_values(
    dataset = dataset, preds_df = SimKGC_results_for_bias_measurement_withoutNAs,
    sensitive_relations = SENSITIVE_RELATIONS)

preds_df_SimKGC_results_for_bias_measurement_withNAs.to_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'preds_df_SimKGC_IB_occupation_9classes_WITHNAs_sensrel_from_entire_v4subset.tsv'),
    sep = '\t', index = False)

preds_df_SimKGC_results_for_bias_measurement_withoutNAs.to_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'preds_df_SimKGC_IB_occupation_9classes_WITHOUTNAs_sensrel_from_entire_v4subset.tsv'),
    sep = '\t', index = False)

# %% CODE CHAPTER: Process KG-BERT evaluation results (KG+LM + FB15K-237)

### IMPORTANT:concatenate all 3 dataframes to a single one (resulting in 1747 rows)

path_to_results = os.path.join(BASE_PATH_HOST, 'results/KG_and_LM/KG-BERT/')
experiment_name = '11.05.2022_16:00_train_and_test_default_FB15K-237_6A100_final'
file_name_1 = 'ranks_with_metadata_test_21.05.2022_18:34_raw_first1426triples.tsv'
file_name_2 = 'ranks_with_metadata_test_25.05.2022_10:37_raw_triples1400-1497.tsv'
file_name_3 = 'ranks_with_metadata_test_22.05.2022_15:28_raw_last300triples.tsv'

raw_file_1_KGBERT = pd.read_csv(os.path.join(path_to_results, experiment_name, file_name_1),
                                sep = '\t', skiprows = 2)
assert len(raw_file_1_KGBERT.columns) == 11
# drop duplicate rows, use head, relation, tail as identifier (unique combination)
# result has 1429 triples
cleaned_file_1_KGBERT = raw_file_1_KGBERT.drop_duplicates(subset = ['head', 'relation', 'tail'],
                                                          keep = 'first', ignore_index = True)

raw_file_2_KGBERT = pd.read_csv(os.path.join(path_to_results, experiment_name, file_name_2),
                                sep = '\t')
assert len(raw_file_2_KGBERT.columns) == 11
# dropping duplicates not necessary here
# but weird last column needs to be removed
# result has 98 triples
cleaned_file_2_KGBERT = raw_file_2_KGBERT.drop('Unnamed: 11', axis = 1, inplace = False)
assert len(cleaned_file_2_KGBERT.columns) == 11

raw_file_3_KGBERT = pd.read_csv(os.path.join(path_to_results, experiment_name, file_name_3),
                                sep = '\t', skiprows = 2)
# drop duplicate rows, use head, relation, tail as identifier (unique combination)
# results in 301 triples
cleaned_file_3_KGBERT = raw_file_3_KGBERT.drop_duplicates(subset = ['head', 'relation', 'tail'],
                                                          keep = 'first', ignore_index = True)
cleaned_file_3_KGBERT = raw_file_3_KGBERT.drop('Unnamed: 11', axis = 1, inplace = False)
assert len(cleaned_file_3_KGBERT.columns) == 11

# merge the 3 files together, again using the head, relation, tail columns
# step 1: concatenate them into a dataframe
KGBERT_results_cleaned = pd.concat(
    [cleaned_file_1_KGBERT, cleaned_file_2_KGBERT, cleaned_file_3_KGBERT])
KGBERT_results_cleaned.drop(['test_triple_index', 'runtime_sec'], axis = 1, inplace = True)
print(f'Length before merging: {len(KGBERT_results_cleaned)}')
# step 2: remove duplicates, using all columns!
# this results in 1821, so it does not work...
KGBERT_results_cleaned.drop_duplicates(keep = 'first')
# step 3: remove duplicates using only head, relation, tail, keeping the first results
# this results in 1747 triples as desired!
KGBERT_results_cleaned_merged = KGBERT_results_cleaned.drop_duplicates(
    subset = ['head', 'relation', 'tail'], keep = 'first', ignore_index = True)

# this returns 1750 rows
KGBERT_results_cleaned.drop_duplicates(subset = ['head', 'relation', 'tail', 'rank_tail'],
                                       keep = 'first')
# this returns 1814 rows
KGBERT_results_cleaned.drop_duplicates(subset = ['head', 'relation', 'tail', 'rank_head'],
                                       keep = 'first')

# It should be 1311 occupation and 436 gender facts = 1747 in total
assert len(KGBERT_results_cleaned_merged) == 1747

# save clean, merged results file, delete the old ones
# KGBERT_results_cleaned_merged.to_csv(os.path.join(path_to_results, experiment_name,
#                                                   'ranks_with_metadata_test_merged_only_gender_occupation_29052022.tsv'),
#                                      sep = '\t', index = False)

TARGET_RELATIONS = ['/people/person/profession']
# profession: 1311 facts

### IMPORTANT: choose only the occupation relations
KGBERT_results_filtered_for_target_mask = KGBERT_results_cleaned_merged['relation'].isin(
    TARGET_RELATIONS)
KGBERT_results_filtered_for_target = KGBERT_results_cleaned_merged[
    KGBERT_results_filtered_for_target_mask].copy()

assert len(KGBERT_results_filtered_for_target) == 1311

# create 'pred_tail_id' column: extract top-1 predictions from top-10 columns
function_to_apply = lambda row: row.strip('[]').split(',')[0].strip("\'")
KGBERT_results_filtered_for_target['pred_tail_id'] = KGBERT_results_filtered_for_target[
    'top_k_entities_IDs_tail'].apply(function_to_apply)

# drop unneeded columns
KGBERT_results_filtered_for_target.drop(
    ['rank_head', 'rank_tail', 'top_k_scores_head', 'top_k_entities_IDs_head', 'top_k_scores_tail',
     'top_k_entities_IDs_tail'], axis = 1, inplace = True)

# rename columns using convention from COLUMN_NAMES_PREDS_DF
COLUMN_NAMES_PREDS_DF
KGBERT_results_filtered_for_target.rename(
    columns = {'head': 'head_id', 'relation': 'relation_id', 'tail': 'true_tail_id'},
    inplace = True)

### IMPORTANT: retrieve natural language labels for all IDs (14951 labels)
entity_to_label_FB15k237 = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                    'data/processed/files_per_model/KG_and_LM_KGBERT_FB15K-237/entity2text.txt'),
                                       sep = '\t', names = ['ID', 'label'])
# add label for relation manually
# --> profession
KGBERT_results_filtered_for_target['relation_label'] = 'profession'

# add the labels for the true tails
KGBERT_results_filtered_for_target = pd.merge(left = KGBERT_results_filtered_for_target,
                                              right = entity_to_label_FB15k237, how = 'left',
                                              left_on = 'true_tail_id', right_on = 'ID')
KGBERT_results_filtered_for_target.rename(columns = {'label': 'true_tail_label'}, inplace = True)
KGBERT_results_filtered_for_target.drop('ID', axis = 1, inplace = True)

# add the labels for the predicted tails
KGBERT_results_filtered_for_target = pd.merge(left = KGBERT_results_filtered_for_target,
                                              right = entity_to_label_FB15k237, how = 'left',
                                              left_on = 'pred_tail_id', right_on = 'ID')
KGBERT_results_filtered_for_target.rename(columns = {'label': 'pred_tail_label'}, inplace = True)
KGBERT_results_filtered_for_target.drop('ID', axis = 1, inplace = True)

# reorder the columns
KGBERT_results_for_bias_measurement = KGBERT_results_filtered_for_target[COLUMN_NAMES_PREDS_DF]

### IMPORTANT: create numeric class encodings for occupation
file_for_class_encoding = 'src/bias_measurement/link_prediction_bias/' \
                          'target_relation_encodings/profession_FB15k237_9classes_using_testset.tsv'
target_relation_class_encoding = pd.read_csv(os.path.join(BASE_PATH_HOST, file_for_class_encoding),
                                             sep = '\t', usecols = [0, 1, 2, 3, 4])
target_relation_class_encoding = target_relation_class_encoding.convert_dtypes()

# retrieve the numeric class labels for 'true_tail_id' and 'pred_tail_id' using merge()
KGBERT_results_for_bias_measurement = pd.merge(left = KGBERT_results_for_bias_measurement,
                                               right = target_relation_class_encoding[
                                                   ['tail_entity_id',
                                                    'class_label_geq50_based_on_testset']],
                                               how = 'left', left_on = 'true_tail_id',
                                               right_on = 'tail_entity_id')
KGBERT_results_for_bias_measurement.rename(
    columns = {'class_label_geq50_based_on_testset': 'true_tail_class_label'}, inplace = True)
KGBERT_results_for_bias_measurement.drop('tail_entity_id', axis = 1, inplace = True)
KGBERT_results_for_bias_measurement = pd.merge(left = KGBERT_results_for_bias_measurement,
                                               right = target_relation_class_encoding[
                                                   ['tail_entity_id',
                                                    'class_label_geq50_based_on_testset']],
                                               how = 'left', left_on = 'pred_tail_id',
                                               right_on = 'tail_entity_id')
KGBERT_results_for_bias_measurement.rename(
    columns = {'class_label_geq50_based_on_testset': 'pred_tail_class_label'}, inplace = True)
KGBERT_results_for_bias_measurement.drop('tail_entity_id', axis = 1, inplace = True)

# reorder columns
KGBERT_results_for_bias_measurement = KGBERT_results_for_bias_measurement[
    ['head_id', 'relation_id', 'relation_label', 'true_tail_id', 'true_tail_label',
     'true_tail_class_label', 'pred_tail_id', 'pred_tail_label', 'pred_tail_class_label']]

### IMPORTANT count and exclude rows where predicted entity is not an occupation tail entity!
# this contains 593 rows
NAs_KGBERT_pred_tail_class_label = KGBERT_results_for_bias_measurement[
    KGBERT_results_for_bias_measurement['pred_tail_class_label'].isnull()]
# this is empty (as expected)
NAs_KGBERT_true_tail_class_label = KGBERT_results_for_bias_measurement[
    KGBERT_results_for_bias_measurement['true_tail_class_label'].isnull()]

# dataframe including all 1311 occupation facts
KGBERT_results_for_bias_measurement_withNAs = KGBERT_results_for_bias_measurement.copy()
# dataframe filtered for NAs with 718 rows
KGBERT_results_for_bias_measurement_withoutNAs = KGBERT_results_for_bias_measurement.drop(
    index = NAs_KGBERT_pred_tail_class_label.index)

### IMPORTANT: add sensitive attribute information as column

SENSITIVE_RELATIONS = ['/people/person/gender']
from pykeen.datasets import FB15k237

dataset = FB15k237()

# GENDER TAIL COUNTS: {'/m/05zppz': 1108, '/m/02zsn': 203}
preds_df_KGBERT_results_for_bias_measurement_withNAs = add_sensitive_relation_values(
    dataset = dataset, preds_df = KGBERT_results_for_bias_measurement_withNAs,
    sensitive_relations = SENSITIVE_RELATIONS)
# GENDER TAIL COUNTS: {'/m/05zppz': 628, '/m/02zsn': 90}
preds_df_KGBERT_results_for_bias_measurement_withoutNAs = add_sensitive_relation_values(
    dataset = dataset, preds_df = KGBERT_results_for_bias_measurement_withoutNAs,
    sensitive_relations = SENSITIVE_RELATIONS)

preds_df_KGBERT_results_for_bias_measurement_withNAs.to_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'preds_df_KG-BERT_occupation_9classes_WITHNAs_sensrel_from_entire_FB15k-237.tsv'),
    sep = '\t', index = False)

preds_df_KGBERT_results_for_bias_measurement_withoutNAs.to_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'preds_df_KG-BERT_occupation_9classes_WITHOUTNAs_sensrel_from_entire_FB15k-237.tsv'),
    sep = '\t', index = False)

# %% CODE CHAPTER Process Rossi evaluation results file (KG only + FB15K-237)

# The basis for this models are files where the predictions of the model on the testset are published.

# count whether I also get 1747 facts when filtering for occupation

path_to_results = os.path.join(BASE_PATH_HOST, 'results/KG_only')
experiment_name = 'Rossi_models_FB15K-237'
file_name = 'RotatE/rotate_filtered_details.csv'
# 'TransE/transe_filtered_details.csv'
# 'DistMult/distmult_filtered_details.csv'
# 'RotatE/rotate_filtered_details.csv'

### IMPORTANT: clean the results file
raw_file_Rossi_FB15k237 = pd.read_csv(os.path.join(path_to_results, experiment_name, file_name),
                                      sep = ';', usecols = [0, 1, 2, 3, 4],
                                      names = ['head_id', 'relation_id', 'true_tail_id',
                                               'prediction_type', 'pred_tail_id'])

# keep only the tail predictions (20,438 rows)
raw_file_Rossi_FB15k237 = raw_file_Rossi_FB15k237[
    raw_file_Rossi_FB15k237['prediction_type'] == 'predict tail'].copy()
assert len(raw_file_Rossi_FB15k237['prediction_type'].unique()) == 1
raw_file_Rossi_FB15k237.drop(columns = 'prediction_type', inplace = True)

# remove brackets from column 'pred_tail_id'
raw_file_Rossi_FB15k237['pred_tail_id'] = raw_file_Rossi_FB15k237['pred_tail_id'].apply(
    lambda row: row.strip('[]'))

### IMPORTANT: choose only the occupation relations
TARGET_RELATIONS = ['/people/person/profession']

Rossi_FB15k237_results_filtered_for_target_mask = raw_file_Rossi_FB15k237['relation_id'].isin(
    TARGET_RELATIONS)
Rossi_FB15k237_results_filtered_for_target = raw_file_Rossi_FB15k237[
    Rossi_FB15k237_results_filtered_for_target_mask].copy()
# reset the index
Rossi_FB15k237_results_filtered_for_target.reset_index(drop = True, inplace = True)

assert len(Rossi_FB15k237_results_filtered_for_target) == 1311

### IMPORTANT: retrieve natural language labels for all IDs (14951 labels)
entity_to_label_FB15k237 = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                    'data/processed/files_per_model/KG_and_LM_KGBERT_FB15K-237/entity2text.txt'),
                                       sep = '\t', names = ['ID', 'label'])
# add label for relation manually
# --> profession
Rossi_FB15k237_results_filtered_for_target['relation_label'] = 'profession'

# add the labels for the true tails
Rossi_FB15k237_results_filtered_for_target = pd.merge(
    left = Rossi_FB15k237_results_filtered_for_target, right = entity_to_label_FB15k237,
    how = 'left', left_on = 'true_tail_id', right_on = 'ID')
Rossi_FB15k237_results_filtered_for_target.rename(columns = {'label': 'true_tail_label'},
                                                  inplace = True)
Rossi_FB15k237_results_filtered_for_target.drop('ID', axis = 1, inplace = True)

# add the labels for the predicted tails
Rossi_FB15k237_results_filtered_for_target = pd.merge(
    left = Rossi_FB15k237_results_filtered_for_target, right = entity_to_label_FB15k237,
    how = 'left', left_on = 'pred_tail_id', right_on = 'ID')
Rossi_FB15k237_results_filtered_for_target.rename(columns = {'label': 'pred_tail_label'},
                                                  inplace = True)
Rossi_FB15k237_results_filtered_for_target.drop('ID', axis = 1, inplace = True)

# reorder the columns
Rossi_FB15k237_results_for_bias_measurement = Rossi_FB15k237_results_filtered_for_target[
    COLUMN_NAMES_PREDS_DF].copy()

### IMPORTANT: create numeric class encodings for occupation
file_for_class_encoding = 'src/bias_measurement/link_prediction_bias/' \
                          'target_relation_encodings/profession_FB15k237_9classes_using_testset.tsv'
target_relation_class_encoding = pd.read_csv(os.path.join(BASE_PATH_HOST, file_for_class_encoding),
                                             sep = '\t', usecols = [0, 1, 2, 3, 4])
target_relation_class_encoding = target_relation_class_encoding.convert_dtypes()

# retrieve the numeric class labels for 'true_tail_id' and 'pred_tail_id' using merge()
Rossi_FB15k237_results_for_bias_measurement = pd.merge(
    left = Rossi_FB15k237_results_for_bias_measurement, right = target_relation_class_encoding[
        ['tail_entity_id', 'class_label_geq50_based_on_testset']], how = 'left',
    left_on = 'true_tail_id', right_on = 'tail_entity_id')
Rossi_FB15k237_results_for_bias_measurement.rename(
    columns = {'class_label_geq50_based_on_testset': 'true_tail_class_label'}, inplace = True)
Rossi_FB15k237_results_for_bias_measurement.drop('tail_entity_id', axis = 1, inplace = True)
Rossi_FB15k237_results_for_bias_measurement = pd.merge(
    left = Rossi_FB15k237_results_for_bias_measurement, right = target_relation_class_encoding[
        ['tail_entity_id', 'class_label_geq50_based_on_testset']], how = 'left',
    left_on = 'pred_tail_id', right_on = 'tail_entity_id')
Rossi_FB15k237_results_for_bias_measurement.rename(
    columns = {'class_label_geq50_based_on_testset': 'pred_tail_class_label'}, inplace = True)
Rossi_FB15k237_results_for_bias_measurement.drop('tail_entity_id', axis = 1, inplace = True)

# reorder columns
Rossi_FB15k237_results_for_bias_measurement = Rossi_FB15k237_results_for_bias_measurement[
    ['head_id', 'relation_id', 'relation_label', 'true_tail_id', 'true_tail_label',
     'true_tail_class_label', 'pred_tail_id', 'pred_tail_label', 'pred_tail_class_label']]

### IMPORTANT count and exclude rows where predicted entity is not an occupation tail entity!
# this is empty for TransE + Distmult (means that the model is actually very good)
# for RotatE there is one NA row
NA_rows_Rossi_FB15k237_pred_tail_class_label = Rossi_FB15k237_results_for_bias_measurement[
    Rossi_FB15k237_results_for_bias_measurement['pred_tail_class_label'].isnull()]
# this is empty for all 3 models (as expected)
NA_rows_Rossi_FB15k237_true_tail_class_label = Rossi_FB15k237_results_for_bias_measurement[
    Rossi_FB15k237_results_for_bias_measurement['true_tail_class_label'].isnull()]

### IMPORTANT: add sensitive attribute information as column

SENSITIVE_RELATIONS = ['/people/person/gender']
from pykeen.datasets import FB15k237

dataset = FB15k237()

if 'RotatE' in file_name:
    # dataframe including all 1311 occupation facts
    Rossi_FB15k237_results_for_bias_measurement_withNAs = Rossi_FB15k237_results_for_bias_measurement.copy()
    # dataframe filtered for the single NA with 1310 rows
    Rossi_FB15k237_results_for_bias_measurement_withoutNAs = Rossi_FB15k237_results_for_bias_measurement.drop(
        index = NA_rows_Rossi_FB15k237_pred_tail_class_label.index)
    # GENDER TAIL COUNTS: {'/m/05zppz': 1108, '/m/02zsn': 203}
    preds_df_Rossi_FB15k237_results_for_bias_measurement_withNAs = add_sensitive_relation_values(
        dataset = dataset, preds_df = Rossi_FB15k237_results_for_bias_measurement_withNAs,
        sensitive_relations = SENSITIVE_RELATIONS)
    # GENDER TAIL COUNTS: {'/m/05zppz': 1107, '/m/02zsn': 203}
    preds_df_Rossi_FB15k237_results_for_bias_measurement_withoutNAs = add_sensitive_relation_values(
        dataset = dataset, preds_df = Rossi_FB15k237_results_for_bias_measurement_withoutNAs,
        sensitive_relations = SENSITIVE_RELATIONS)

    preds_df_Rossi_FB15k237_results_for_bias_measurement_withNAs.to_csv(
        os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                     'preds_df_Rossi_RotatE_FB15k237_occupation_9classes_WITHNAs_sensrel_from_entire_FB15k-237.tsv'),
        sep = '\t', index = False)

    preds_df_Rossi_FB15k237_results_for_bias_measurement_withoutNAs.to_csv(
        os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                     'preds_df_Rossi_RotatE_FB15k237_occupation_9classes_WITHOUTNAs_sensrel_from_entire_FB15k-237.tsv'),
        sep = '\t', index = False)

# GENDER TAIL COUNTS: {'/m/05zppz': 1108, '/m/02zsn': 203}
preds_df_Rossi_FB15k237_results_for_bias_measurement_withputanyNAs = add_sensitive_relation_values(
    dataset = dataset, preds_df = Rossi_FB15k237_results_for_bias_measurement,
    sensitive_relations = SENSITIVE_RELATIONS)

preds_df_Rossi_FB15k237_results_for_bias_measurement_withputanyNAs.to_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'preds_df_Rossi_DistMult_FB15k237_occupation_9classes_DOESNOTHAVENAs_sensrel_from_entire_FB15k-237.tsv'),
    sep = '\t', index = False)

# %% Code chapter: Try out using scikit-learn for calculating fairness metrics

# average = None: Calculate and output metric per label, output is a list
# average = 'macro': Calculate metrics for each label, and find their unweighted mean.
# This does not take label imbalance into account. Get same result by averaging "None" results.
# average = 'micro': Calculate metrics globally by counting the total true positives,
# false negatives and false positives.
# average = 'weighted': Calculate metrics for each label, and find their average weighted
# by support (the number of true instances for each label). This alters ‘macro’ to account
# for label imbalance; it can result in an F-score that is not between precision and recall.


file_name = 'preds_df_SimKGC_sensitive_relations_from_entire_v4subset.tsv'
experiment_name = experiment_name

preds_df = pd.read_csv(os.path.join(path_to_results, experiment_name, file_name), sep = '\t')

assert preds_df[preds_df['pred_tail_class_label'].isnull()].empty
assert preds_df[preds_df['true_tail_class_label'].isnull()].empty

# TODO adapt labels argument to number of classes
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report, precision_score

precision_macro = precision_score(
    y_true = preds_df['true_tail_class_label'].to_numpy(dtype = 'int'),
    y_pred = preds_df['pred_tail_class_label'].to_numpy(dtype = 'int'), average = 'macro',
    labels = [*range(0, 9, 1)])

precision_micro = precision_score(
    y_true = preds_df['true_tail_class_label'].to_numpy(dtype = 'int'),
    y_pred = preds_df['pred_tail_class_label'].to_numpy(dtype = 'int'), average = 'micro',
    labels = [*range(0, 9, 1)])

precision_none = precision_score(y_true = preds_df['true_tail_class_label'].to_numpy(dtype = 'int'),
                                 y_pred = preds_df['pred_tail_class_label'].to_numpy(dtype = 'int'),
                                 average = None, labels = [*range(0, 9, 1)])
assert precision_macro == np.mean(precision_none)
print(precision_micro)
print(precision_macro)
print(precision_none)

precision, recall, fscore, support = precision_recall_fscore_support(
    y_true = preds_df['true_tail_class_label'].to_numpy(dtype = 'int'),
    y_pred = preds_df['pred_tail_class_label'].to_numpy(dtype = 'int'), average = 'macro',
    labels = [*range(0, 9, 1)])

test = confusion_matrix(y_true = preds_df['true_tail_class_label'].to_numpy(dtype = 'int'),
                        y_pred = preds_df['pred_tail_class_label'].to_numpy(dtype = 'int'),
                        normalize = 'all').ravel()

report = classification_report(y_true = preds_df['true_tail_class_label'].to_numpy(dtype = 'int'),
                               y_pred = preds_df['pred_tail_class_label'].to_numpy(dtype = 'int'))
print(report)

ConfusionMatrixDisplay.from_predictions(
    y_true = preds_df['true_tail_class_label'].to_numpy(dtype = 'int'),
    y_pred = preds_df['pred_tail_class_label'].to_numpy(dtype = 'int'), normalize = 'all',
    include_values = False)

plt.show()

# preds_df.to_csv(
#     os.path.join(path_to_results, experiment_name, f'preds_df_SimKGC_IB_250522_v2_9occ.tsv'),
#     indßex = False, sep = '\t')

# TODO Try out calculating metrics using Keidar code
preds_df = preds_df.drop([], axis = 1)


# TODO add sensitive attribute information from the dataset

# %% Code chapter: reuse code from my research project

# general idea: I also had a multiclass-classification scenario there, so it's the same!

def calculate_evaluation_metrics(labels, predictions):
    """
    Parameters
    ----------
    labels : torch.Tensor
    predictions : torch.Tensor
    """
    if predictions.size() != labels.size():
        raise ValueError("Labels and prediction tensors must be of same length!")

    # precision, recall, f1 score and support per class
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true = labels,
                                                                           y_pred = predictions,
                                                                           average = None, beta = 1)
    # confusion matrix
    confuse_matrix = confusion_matrix(y_true = labels, y_pred = predictions)

    # calculations from: https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
    # true positives = diagonal of the confusion matrix (as many elements as classes)
    TP = np.diag(confuse_matrix)
    # predictions per class = confuse_matrix.sum(axis = 0)
    # samples per class (true labels) = confuse_matrix.sum(axis = 1)

    # false positives = samples predicted wrongly predicted as class X
    # --> subtract true positives from samples per class predicted as that class
    FP = confuse_matrix.sum(axis = 0) - TP
    # false negatives = the number of samples that truly belong to a specific class, but were not classifies as that class
    # --> subtract the true positives from the number of true samples (= support) per class
    FN = confuse_matrix.sum(axis = 1) - TP
    # true negatives = all samples that are left over
    # all elements of the -ith position in the four vectors will sum up to the number of samples
    # e.g. TP[4]+FP[4]+FN[4]+TN[4] = number_of_samples
    TN = confuse_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # false negative rate (fraction of false negatives of all truly positive examples)
    FNR = FN / (FN + TP)
    # false positive rate/fall-out (fraction of false positives of all truly negative examples)
    FPR = FP / (FP + TN)

    return {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'support': support,
            'confusion_matrix': confuse_matrix, 'false_negative_rate': FNR,
            'false_positive_rate': FPR}
