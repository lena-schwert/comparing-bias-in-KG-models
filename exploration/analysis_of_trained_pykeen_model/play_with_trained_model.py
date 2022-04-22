# %% Imports and preliminaries

import os
import torch



from src.utils import set_base_path_based_on_host


BASE_PATH_HOST = set_base_path_based_on_host()

# %% load trained model

abs_path_to_model_pkl = os.path.join(BASE_PATH_HOST, 'results/TransE_fullW5M_80epochs/trained_model.pkl')

# RAM usage before: 9.7 GB
model = torch.load(abs_path_to_model_pkl)
# RAM usage afterwards: not really higher

# load triples factory needed for prediction
from src.CLI_scripts.experiments_KG_only import HumanWikidata5M_pykeen

tf_for_prediction = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                           rel_training_set_path = 'data_preprocessing/training_data_0.8_rs42_06_01_2022_15:58_DEBUGGING.tsv',
                                           rel_validation_set_path = 'data_preprocessing/validation_data_0.1_rs42_06_01_2022_15:58_DEBUGGING.tsv',
                                           rel_test_set_path = 'data_preprocessing/test_data_0.1_rs42_06_01_2022_15:58_DEBUGGING.tsv')



# %% make new predictions using this model

# pykeen doc: https://pykeen.readthedocs.io/en/stable/tutorial/making_predictions.html

from pykeen.models import predict

predicted_tails_df = predict.get_tail_prediction_df(
    model,
    head_label = 'Q31',
    relation_label = 'P31',
    triples_factory = None
)
