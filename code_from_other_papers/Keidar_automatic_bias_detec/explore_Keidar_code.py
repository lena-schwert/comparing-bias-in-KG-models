import os
import sys


os.getcwd()
os.chdir('/home/lena/git/master_thesis_bias_in_NLP/code_from_other_papers/Keidar_automatic_bias_detec')
sys.path.append('/home/lena/git/master_thesis_bias_in_NLP/code_from_other_papers/Keidar_automatic_bias_detec')

import pandas as pd
from pykeen.datasets import FB15k237

# import from repo's files
from visualization import preds_histogram

# %% Take a look at the pred_df

pred_df_transe = pd.read_csv('preds_dfs/preds_df_transe.csv')

preds_histogram(pred_df_transe)


# %% Import their list of human entities + relations

human_ents = pd.read_pickle('data/wiki5m/human_ent_rel_sorted_list.pkl')


# %% Try to understand preds_df by manually mapping IDs

dataset_man = FB15k237()
# first five entity to ID mapping:
# {'/m/010016': 0, '/m/0100mt': 1, '/m/0102t4': 2, '/m/0104lr': 3, '/m/0105y2': 4, '/m/0106dv': 5,

# first three relation to ID mappings:
# '/american_football/football_team/current_roster./sports/sports_team_roster/position': 0,
# '/award/award_category/category_of': 1,
# '/award/award_category/disciplines_or_subjects': 2, '/award/award_category/nominees./award/award_nomination/nominated_for': 3

# MANUAL MAPPING FROM PYKEEN IDs to Freebase stuff

# gender
# 8663
# 5420



def search_keys_by_val(dict, byVal):
    keysList = []
    itemsList = dict.items()
    for item in itemsList:
        if item[1] == byVal:
            keysList.append(f'key: {item[0]}, value: {byVal})')
    return keysList

def search_val_by_key(dict, byKey):
    valList = []
    itemsList = dict.items()
    for item in itemsList:
        if item[0] == byKey:
            valList.append(f'key: {item[0]}, value: {byKey})')
    return valList


