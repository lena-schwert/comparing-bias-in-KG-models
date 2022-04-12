# %% Imports

import os
import pickle
import pandas as pd
from pykeen.models import ComplEx, TransE, DistMult, RotatE
from easydict import EasyDict as edict
import torch

from src.utils import set_base_path_based_on_host

BASE_PATH_HOST = set_base_path_based_on_host()


# This merges multiple files taken from the mianzg/kgbasdetec repo
#   /wiki5m/process_wiki5m.py
#   /wiki5m/wrap_wiki5m.py

def load_embeddings(embedding_path: str, return_human_head_ents: bool = False):
    """
    Given a path, loads the entity/relation2id files and the embeddings.
    Meant for use on the graphvite pretrained models (.pkl files).

    Parameters
    ----------
    return_human_head_ents
    embedding_path

    Returns
    -------
    (entity2id, relation2id): EasyDict = mapping from Q/PID to numeric ID
    (entity_embeddings, relation_embeddings): EasyDict = arrays of embeddings (vectors of size 512)
    """
    import pickle
    with open(embedding_path, "rb") as f:
        model = pickle.load(f)
    entity2id = model.graph.entity2id
    relation2id = model.graph.relation2id
    # Note that embeddings have dimension 512
    entity_embeddings = model.solver.entity_embeddings
    relation_embeddings = model.solver.relation_embeddings
    print("Num Entities in Graphvite: ", len(entity2id.keys()))
    print("Num Relations in Graphvite: ", len(relation2id.keys()))
    print('Embedding dimension: 512')

    if return_human_head_ents:
        # Exploration: Do all entities and relations of HumanWikidata5M have pre-trained embeddings?
        set_of_graphvite_entities = set(model.graph.id2entity)
        with open(os.path.join(BASE_PATH_HOST,
                               'data/interim/wikidata5m_human_entities_01042022_v2.tsv'), 'r') as f:
            HumanWikidata5M_humans = set(f.read().splitlines())
        f.close()

        # find the entities that are in my human entities, but that do not have a graphvite embedding
        # A.difference(B) = all elements that are in A but not in B
        # 39 QIDs are head entities in my human subset, but not contained in the pre-trained Graphvite model
        humans_without_embedding = HumanWikidata5M_humans.difference(set_of_graphvite_entities)
        humans_with_embedding = HumanWikidata5M_humans.intersection(set_of_graphvite_entities)
        assert len(humans_without_embedding) + len(humans_with_embedding) == len(
            HumanWikidata5M_humans)

        # Create a special set of entities and facts to be used with Graphvite embeddings

        # filter out all the facts where the head entity does not have a Graphvite embedding
        human_triples_v3 = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                    'data/interim/wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv'),
                                       sep = '\t',
                                       names = ['head_entity', 'relation', 'tail_entity'])

        # This basically corresponds to Step 6 in wikidata5m_preprocessing.py
        humans_without_embedding_list = list(humans_without_embedding)
        human_triples_to_remove = human_triples_v3.query(
            'head_entity == @humans_without_embedding_list')
        # This dataframe has 78 rows that will be excluded.

        # The remaining human subset contains 11,114,719 triples.
        human_triples_graphvite = human_triples_v3.drop(index = human_triples_to_remove.index)

        folder_to_save = os.path.join(BASE_PATH_HOST, 'data/interim/KG_only_files')

        # save to TSV file
        human_triples_graphvite.to_csv(os.path.join(folder_to_save,
                                                    'wikidata5m_human_facts_for_graphvite_models_08042022_v1.tsv'),
                                       sep = '\t', header = False, index = False)

        ### entities.txt: list of all remaining 1503452 entities in the dataset
        with open(
                os.path.join(folder_to_save, 'human_entities_for_graphvite_models_08042022_v1.txt'),
                'w') as f:
            f.writelines(''.join([str(x) + '\n' for x in humans_with_embedding]))
        f.close()

        print('Wrote all new files to disk.')

    return entity2id, entity_embeddings


def convert2humanembeddings(embedding_name, entities_to_extract):
    """

    Parameters
    ----------
    embedding_name: String that depicts the model name. The corresponding file needs to exist already.

    Returns
    -------
    saves the human head entity embeddings to disk
    """
    model_path = os.path.join(folder_for_loading_and_saving_models,
                              f'{embedding_name}_wikidata5m.pkl')
    ent2id, entity_embeddings = load_embeddings(model_path)
    entities_to_extract = list(entities_to_extract)
    # retrieve numeric IDs for all desired entities and relations
    human_ent_ids = [ent2id[i] for i in entities_to_extract]
    # based on the numeric IDs, extract the corresponding embedding vectors
    human_entity_embeddings = entity_embeddings[human_ent_ids]
    # Create a dict of string ID to embedding
    wikidata_ID_to_entity_embedding = dict(zip(entities_to_extract, human_entity_embeddings))

    # doublecheck for the first entry
    entities_to_extract[0]  # Q6632165
    ent2id.get('Q6632165')  # 2821764
    import numpy as np
    assert np.all(entity_embeddings[2821764] == wikidata_ID_to_entity_embedding.get('Q6632165'))

    # Store human embeddings in a re-identifiable format

    # needed properties: model.embedding_dim, model.entity_embeddings(numeric IDs)
    # IMPORTANT: first iteration, simply use an EasyDict

    # model = edict({
    #     'embedding_dim': human_entity_embeddings.shape[1],  # dim = 512
    #     # TODO make the entity embeddings retraceable using numeric IDs!
    #     'entity_embeddings': None
    # })

    file_path = os.path.join(folder_for_loading_and_saving_models,
                             f'{embedding_name}_pretrained_human_W5M_entity_embeddings_dict.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(wikidata_ID_to_entity_embedding, f, protocol = pickle.HIGHEST_PROTOCOL)

    # this is similar to /wiki5m/wrap_wiki5m.py

    from src.utils import HumanWikidata5M_pykeen
    dataset = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,
                                     rel_training_set_path = 'data/interim/KG_only_files/train.tsv',
                                     rel_validation_set_path = 'data/interim/KG_only_files/validation.tsv',
                                     rel_test_set_path = 'data/interim/KG_only_files/test.tsv')

    # class WIKI5M_TransE(TransE):
    #     def __init__(self, dataset, model_path = "trained_models/KG_only/graphvite_pretrained_W5M/transe_pretrained_human_W5M.pkl"):
    #         with open(os.path.join(BASE_PATH_HOST, model_path), "rb") as fin:
    #             entity_embeddings = pickle.load(fin)
    #             embedding_dim = entity_embeddings.shape[1]
    #
    #         # arguments given to superclasses can be found in
    #         # pykeen.TransE, trans_e.py, line 55
    #         # pykeen.EntityRelationEmbeddingModel, base.py, line 690
    #         super(WIKI5M_TransE, self).__init__(triples_factory = dataset.training,
    #                                             random_seed = 42,
    #                                             embedding_dim = embedding_dim)
    #         self.entity_embeddings._embeddings = torch.from_numpy(
    #              entity_embeddings)
    #         # self.relation_embeddings.weight.data = torch.from_numpy(relation_embeddings)
    #
    # model_wrapped = WIKI5M_TransE(dataset = dataset)



    # with open(file_path, 'rb') as handle:  #     b = pickle.load(handle)

    # from src.utils import HumanWikidata5M_pykeen  # HumanWikidata5M = dataset = HumanWikidata5M_pykeen(base_path_host = BASE_PATH_HOST,  #                                                    rel_training_set_path = 'data/interim/training_data_0.8_rs42_06_04_2022_18:10.tsv',  #                                                    rel_validation_set_path = 'data/interim/validation_data_0.1_rs42_06_04_2022_18:10.tsv',  #                                                    rel_test_set_path = 'data/interim/test_data_0.1_rs42_06_04_2022_18:10.tsv')

    # # Should have the same properties as a pykeen model  # import torch  # pykeen_model = torch.load(os.path.join(BASE_PATH_HOST,  #                                        'results/KG_only/TransE_fullW5M_80epochs/checkpoints/checkpoint_TransE_fullW5M_80epochs.pt'),  #                           map_location = torch.device('cpu'))  # class WIKI5M_TransE(TransE):  #     def __init__(self, dataset):  #         embedding_dim = human_entity_embeddings.shape[1]  # dim = 512  #  #         super(WIKI5M_TransE, self).__init__(embedding_dim)  #  # test = WIKI5M_TransE(dataset = HumanWikidata5M)

    # original code for saving with the code above  # pickle.dump((human_entity_embeddings,), os.path.join(folder_for_loading_and_saving_models,  #                                                      f'{embedding_name}_pretrained_human_W5M.pkl'))


def wrap_graphvite_embeddings_as_pykeen_models():
    # refer to: /wiki5m/wrap_wiki5m.py
    pass


# %% For each pretrained model, extract and save the human embeddings only

# As we deal with the subset of human facts, we don't need all embeddings.
# Keep only the embeddings for relations + entities that are part of the
# human facts subset.

# uses files:
#   trained_models/KG_only/graphvite_pretrained_W5M/transe_wikidata5m.pkl
#   data/interim/KG_only_files/human_entities_for_graphvite_models_08042022_v1.txt
#   data/processed/relations_04042022_v1.txt

# creates files:
#   transe_pretrained_human_W5M.pkl

embeddings = ["transe"]  # "transe", "distmult", "complex", "rotate"
folder_for_loading_data = os.path.join(BASE_PATH_HOST, 'data/processed')
folder_for_loading_and_saving_models = os.path.join(BASE_PATH_HOST,
                                                    'trained_models/KG_only/graphvite_pretrained_W5M')

if not os.path.isfile(os.path.join(BASE_PATH_HOST,
                                   'data/interim/KG_only_files/human_entities_for_graphvite_models_08042022_v1.txt')):
    load_embeddings(embedding_path = os.path.join(folder_for_loading_and_saving_models,
                                                  'transe_wikidata5m.pkl'),
                    return_human_head_ents = True)

with open(os.path.join(
        BASE_PATH_HOST, 'data/interim/KG_only_files/human_entities_for_graphvite_models_08042022_v1.txt'),
          'r') as f:
    graphvite_human_entities = set(f.read().splitlines())
f.close()

for e in embeddings:
    convert2humanembeddings(e, graphvite_human_entities)
