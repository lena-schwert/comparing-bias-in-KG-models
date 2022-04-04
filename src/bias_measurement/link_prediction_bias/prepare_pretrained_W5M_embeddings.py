# %% Imports

import os
import pickle
from src.utils import set_base_path_based_on_host

BASE_PATH_HOST = set_base_path_based_on_host()


# This merges multiple files taken from the mianzg/kgbasdetec repo
#   /wiki5m/process_wiki5m.py

def load_embeddings(embedding_path):
    """
    Given a path, loads the entity/relation2id files and the embeddings.
    Meant for use on the graphvite pretrained models (.pkl files).

    Parameters
    ----------
    embedding_path

    Returns
    -------
        two tuples, one containing the ID mappings, one the embeddings
    """
    with open(embedding_path, "rb") as f:
        model = pickle.load(f)
    entity2id = model.graph.entity2id
    relation2id = model.graph.relation2id
    entity_embeddings = model.solver.entity_embeddings
    relation_embeddings = model.solver.relation_embeddings
    print("Num Entities: ", len(entity2id.keys()))
    print("Num Relations: ", len(relation2id.keys()))
    return (entity2id, relation2id), (entity_embeddings, relation_embeddings)


def convert2humanembeddings(embedding_name, entlist, rellist):
    """

    Parameters
    ----------
    embedding_name: String that depicts the model name:
    entlist: list of QID strings (= entities) for which pretrained embeddings will be extracted
    rellist: list of PID strings (= relations) for which pretrained embeddings will be extracted

    Returns
    -------
    saves the human embeddings to disk
    """
    model_path = os.path.join(folder_for_loading_and_saving_models, f'{embedding_name}_wikidata5m.pkl')
    (ent2id, rel2id), (entity_embeddings, relation_embeddings) = load_embeddings(model_path)
    human_ent_ids = [ent2id[i] for i in entlist]
    human_rel_ids = [rel2id[i] for i in rellist]
    human_entity_embeddings = entity_embeddings[human_ent_ids]
    human_relation_embeddings = relation_embeddings[human_rel_ids]
    pickle.dump((human_entity_embeddings, human_relation_embeddings),
                os.path.join(folder_for_loading_and_saving_models, f'{embedding_name}_pretrained_human_W5M.pkl'))


# %% For each pretrained model, extract and save the human embeddings only

# As we deal with the subset of human facts, we don't need all embeddings.
# Keep only the embeddings for relations + entities that are part of the
# human facts subset.

# uses files:
#   trained_models/KG_only/graphvite_pretrained_W5M/transe_wikidata5m.pkl
#   data/processed/entities_04042022_v1.txt
#   data/processed/relations_04042022_v1.txt

# creates files:
#   transe_pretrained_human_W5M.pkl

embeddings = ["transe"]  # "transe", "distmult", "complex", "rotate"
folder_for_loading_data = os.path.join(BASE_PATH_HOST, 'data/processed')
folder_for_loading_and_saving_models = os.path.join(BASE_PATH_HOST, 'trained_models/KG_only/graphvite_pretrained_W5M')
# (will be a list of strings)
with open(os.path.join(folder_for_loading_data, 'entities_04042022_v1.txt'), 'r') as file_ent:
    entlist = file_ent.read().splitlines()
file_ent.close()
with open(os.path.join(folder_for_loading_data, 'relations_04042022_v1.txt'), 'r') as file_rel:
    rellist = file_rel.read().splitlines()
file_rel.close()

for e in embeddings:
    convert2humanembeddings(e, entlist, rellist)
