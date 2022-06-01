 
# %% Imports
# In-built modules
import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pykeen
import torch
from tqdm import tqdm

# Internal Imports
from src.utils import set_base_path_based_on_host, get_triples_df, HumanWikidata5M_pykeen
from src.bias_measurement.link_prediction_bias.utils import get_sensitive_and_target_relations

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 750)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 20)

BASE_PATH_HOST = set_base_path_based_on_host()



COLUMN_NAMES_PREDS_DF = ['head_id', 'relation_id', 'relation_label', 'true_tail_id',
                         'true_tail_label', 'pred_tail_id', 'pred_tail_label']


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
#occupation_triples_array = occupation_triples.to_numpy(dtype = str)

# alternatively, try to use functions usually used with PipelineResult:
# https://pykeen.readthedocs.io/en/stable/api/pykeen.models.predict.predict_triples_df.html
from pykeen.models.predict import predict_triples_df, get_tail_prediction_df
from pykeen.models import predict

predicted_tail_entity_IDs = []
predicted_tail_entity_labels = []

for rowtuple in occupation_triples.itertuples():
#    print(rowtuple)
    prediction_result = get_tail_prediction_df(model = trained_model, head_label = rowtuple.head_id,
                                               relation_label = rowtuple.relation_id,
                                               triples_factory = dataset_HumanWikidata5M.training,
                                               testing = dataset_HumanWikidata5M.testing.mapped_triples,
                                               remove_known = True)
    highest_score_row = prediction_result.iloc[:1]
    predicted_tail_entity_IDs.append(highest_score_row['tail_label'].item())
    predicted_tail_entity_labels.append(dataset_HumanWikidata5M.entity_numID_to_label.get(highest_score_row['tail_id'].item()))

print('Finished extracting the tail entities')

pykeen_HW5M_results_for_bias_measurement = occupation_triples.copy()

pykeen_HW5M_results_for_bias_measurement['pred_tail_id'] = predicted_tail_entity_IDs
pykeen_HW5M_results_for_bias_measurement['pred_tail_label'] = predicted_tail_entity_labels

pykeen_HW5M_results_for_bias_measurement.to_csv(
    os.path.join(BASE_PATH_HOST, 'results/bias_measurement/link_prediction_bias/',
                 'predsdf_pykeen_TransE_predictions_occupation.tsv'),
    sep = '\t', index = False)
