# %% Imports
# In-built modules
import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

# Internal Imports
from src.utils import set_base_path_based_on_host, get_triples_df, HumanWikidata5M_pykeen
from src.bias_measurement.link_prediction_bias.utils import get_sensitive_and_target_relations

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 750)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 20)

BASE_PATH_HOST = set_base_path_based_on_host()

# %% Set target and sensitive relation(s)

SENSITIVE_RELATIONS, TARGET_RELATIONS = get_sensitive_and_target_relations(
    dataset_name = 'Wikidata5m')

COLUMN_NAMES_PREDS_DF = ['head_id', 'relation_id', 'relation_label', 'true_tail_id',
                         'true_tail_label', 'pred_tail_id', 'pred_tail_label']

# %% TODO Figure out how many classes to use for each sensitive/target relation

# %% TODO extract sensitive attribute information from the v4 subset of HumanWikidata5M


def add_relation_values(dataset, preds_df, bias_relations):
    """
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

    assert type(bias_relations) == list

    # TODO change this, access the triples from all dataset splits! Create a union of all splits

    # access the test triples of the dataset which have the bias relations (as string ID triples)
    bias_relations_triplets_mask = dataset.testing.get_mask_for_relations(bias_relations)
    bias_relations_triplets = dataset.testing.triples[bias_relations_triplets_mask]
    # only select the bias_relation facts for which the human head entity is part of preds_df['entity']
    #bias_relations_triplets = [tr for tr in bias_relations_triplets if dataset.entity_to_id[tr[0]] in preds_df['head_entity'].values]
    entity_to_tail = {}
    # for each bias relation, create a key-value pair, where:
    # key = bias relation string, value = empty dict
    # e.g. {'P21': {}}
    for bias_rel in bias_relations:
        entity_to_tail[bias_rel] = {}
    # for each bias relation triple, add the numeric head and tail ID to entity_to_tail
    # e.g. {'P21': {379209: 1370033, 763948: 1370033}}
    for head, rel, tail in bias_relations_triplets:
        # retrieve numeric ID for head and tail entity
        #head_id = dataset.entity_to_id.get(head)
        #tail_id = dataset.entity_to_id.get(tail)
        # create a dict entry, where key = num ID head, value = num ID tail
        #entity_to_tail[rel][head_id] = tail_id
        # new code
        entity_to_tail[rel][head] = tail
    # for each bias relation, create a column of numeric ID tail values for the corresponding human head entity
    for bias_rel in bias_relations:
        # for each human head_entity in preds_df, retrieve the numeric ID for the corresponding tail
        # value will be -1 if this fact does not exist in the test set
        # TODO maybe retrieve the sensitive relation facts from the entire dataset instead?
        preds_df[bias_rel] = [get_tail(bias_rel, head_entity) for head_entity in preds_df['head_id'].values]
        # count the occurrence
        attr_counts = Counter(preds_df[bias_rel])
        # IMPORTANT: set a threshold for removing facts that are considered too rare
        #preds_df[rel] = preds_df[rel].apply(lambda x: remove_infreq_attributes(attr_counts, x))
    return preds_df


SENSITIVE_RELATIONS = ['P21', 'P27', 'P127', 'P140']
# gender, country of citizenship, ethnic group, religion

rel_training_set_path = 'data/processed/output_of_preprocessing/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv'
rel_validation_set_path = 'data/processed/output_of_preprocessing/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
rel_test_set_path = 'data/processed/output_of_preprocessing/test_data_subset_0.05_rs42_06_05_2022_15:11.tsv'
dataset = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                 rel_training_set_path = rel_training_set_path,
                                 rel_validation_set_path = rel_validation_set_path,
                                 rel_test_set_path = rel_test_set_path)

path_to_results = os.path.join(BASE_PATH_HOST, 'results/KG_and_LM/SimKGC/')
experiment_name = '12.05.2022_19:34_wiki5m_trans_train_SimKGC_IB_4xa6k5'
preds_df_SimKGC = pd.read_csv(
    os.path.join(path_to_results, experiment_name, "preds_df_SimKGC_IB_250522_v2_9occ_correct.tsv"),
    sep = '\t')

preds_df_with_sensitive_relations = add_relation_values(dataset = dataset,
                                                        preds_df = preds_df_SimKGC,
                                                        bias_relations = SENSITIVE_RELATIONS)

# preds_df_with_sensitive_relations.to_csv(os.path.join(path_to_results, experiment_name,
#                                                        f'preds_df_SimKGC_sensitive_relations_from_test_like_Keidar.tsv'),
#                                           index = False, sep = '\t')


# %% TODO extract sensitive attribute information from Fb15k-237

# %% TODO Process SimKGC evaluation results (KG+LM + HumanWikidata5M)

path_to_results = os.path.join(BASE_PATH_HOST, 'results/KG_and_LM/SimKGC/')
experiment_name = '12.05.2022_19:34_wiki5m_trans_train_SimKGC_IB_4xa6k5'
file_name = 'eval_test_data_subset_0.05.tsv.json_forward_model_last.mdl.json'

raw_file_SimKGC = pd.read_json(os.path.join(path_to_results, experiment_name, file_name))

raw_file_SimKGC['relation_id'].value_counts()

SimKGC_results_filtered_for_target_mask = raw_file_SimKGC['relation_id'].isin(TARGET_RELATIONS)
SimKGC_results_filtered_for_target = raw_file_SimKGC[SimKGC_results_filtered_for_target_mask]
# reorder columns
SimKGC_results_filtered_for_target = SimKGC_results_filtered_for_target[
    ['head_id', 'head', 'relation_id', 'relation', 'tail_id', 'tail', 'pred_tail_id', 'pred_tail',
     'topk_scores_labels', 'correct', 'pred_score', 'topk_scores', 'topk_scores_ids', 'rank']]

# SimKGC_results_filtered_for_target.to_csv(os.path.join(path_to_results, experiment_name,
#                                                        f'SimKGC_forward_target_{TARGET_RELATIONS}.tsv'),
#                                           index = False, sep = '\t')

# SimKGC_results_filtered_for_target = pd.read_csv(
#     os.path.join(path_to_results, experiment_name, "SimKGC_forward_target_['P106'].tsv"),
#     sep = '\t')

# TODO automate file name in case this part of the script is wrapped in a function

# recode the tail values into a fixed set of classes
file_for_class_encoding = 'src/bias_measurement/link_prediction_bias/' \
                          'target_relation_encodings/occupation_P106_top9classes_using_v4testset.tsv'
target_relation_class_encoding = pd.read_csv(os.path.join(BASE_PATH_HOST, file_for_class_encoding),
                                             sep = '\t', usecols = [0, 1, 2, 3, 4])
target_relation_class_encoding = target_relation_class_encoding.convert_dtypes()

SimKGC_results_for_bias_measurement = SimKGC_results_filtered_for_target[
    ['head_id', 'relation_id', 'relation', 'tail_id', 'tail', 'pred_tail_id', 'pred_tail']]
SimKGC_results_for_bias_measurement.columns = COLUMN_NAMES_PREDS_DF

# retrieve the numeric class labels for 'true_tail_id' and 'pred_tail_id' using merge()

# TODO aadapt class labels column name automatically
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

# count NAs in the label columns
# this contains 40 rows
NAs_pred_tail_class_label = SimKGC_results_for_bias_measurement[
    SimKGC_results_for_bias_measurement['pred_tail_class_label'].isnull()]
# this is empty
NAs_true_tail_class_label = SimKGC_results_for_bias_measurement[
    SimKGC_results_for_bias_measurement['true_tail_class_label'].isnull()]

# replace any NAs with class 0 = OTHER
SimKGC_results_for_bias_measurement.fillna(value = 0, inplace = True)
# detect dtypes
SimKGC_results_for_bias_measurement = SimKGC_results_for_bias_measurement.convert_dtypes()
# reorder columns
SimKGC_results_for_bias_measurement = SimKGC_results_for_bias_measurement[
    ['head_id', 'relation_id', 'relation_label', 'true_tail_id', 'true_tail_label',
     'true_tail_class_label', 'pred_tail_id', 'pred_tail_label', 'pred_tail_class_label']]

preds_df = None

# TODO Try out calculating metrics using sklearn multiclass metrics
# average = 'macro': Calculate metrics for each label, and find their unweighted mean.
# This does not take label imbalance into account.
# average = 'micro': Calculate metrics globally by counting the total true positives,
# false negatives and false positives.

# TODO adapt labels argument to number of classes
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report

precision, recall, fscore, support = precision_recall_fscore_support(
    y_true = SimKGC_results_for_bias_measurement['true_tail_class_label'].to_numpy(dtype = 'int'),
    y_pred = SimKGC_results_for_bias_measurement['pred_tail_class_label'].to_numpy(dtype = 'int'),
    average = None, labels = [*range(0, 35, 1)])

test = confusion_matrix(
    y_true = SimKGC_results_for_bias_measurement['true_tail_class_label'].to_numpy(dtype = 'int'),
    y_pred = SimKGC_results_for_bias_measurement['pred_tail_class_label'].to_numpy(dtype = 'int'),
    normalize = 'all').ravel()

report = classification_report(
    y_true = SimKGC_results_for_bias_measurement['true_tail_class_label'].to_numpy(dtype = 'int'),
    y_pred = SimKGC_results_for_bias_measurement['pred_tail_class_label'].to_numpy(dtype = 'int'))
print(report)

ConfusionMatrixDisplay.from_predictions(
    y_true = SimKGC_results_for_bias_measurement['true_tail_class_label'].to_numpy(dtype = 'int'),
    y_pred = SimKGC_results_for_bias_measurement['pred_tail_class_label'].to_numpy(dtype = 'int'),
    normalize = 'all', include_values = False)

plt.show()

SimKGC_results_for_bias_measurement.to_csv(
    os.path.join(path_to_results, experiment_name, f'preds_df_SimKGC_IB_250522_v2_9occ.tsv'),
    indßex = False, sep = '\t')

# TODO Try out calculating metrics using Keidar code
preds_df = SimKGC_results_for_bias_measurement.drop([], axis = 1)

# TODO add sensitive attribute information from the dataset



# %% TODO Process pykeen evaluation results (KG only + HumanWikidata5M)

# %% TODO Process KG-BERT evaluation results (KG+LM + FB15K-237)


# TODO merge the 3 weird result files
# concatenate all 3 dataframes to a single one
# then use FB15K237_testset_only_gender_occupation_facts_complete.tsv as left df in a pd.merge
# then there should be 1747 facts overall


# %% TODO Process Rossi evaluation results file (KG only + FB15K-237)

# count whether I also get 1747 facts when filtering for occupation and
