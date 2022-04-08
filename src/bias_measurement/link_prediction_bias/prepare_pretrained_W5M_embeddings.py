# %% Imports

import os
import pickle
import pandas as pd

from src.utils import set_base_path_based_on_host

BASE_PATH_HOST = set_base_path_based_on_host()


# This merges multiple files taken from the mianzg/kgbasdetec repo
#   /wiki5m/process_wiki5m.py

def load_embeddings(embedding_path: str, return_human_ents_rels: bool = False):
    """
    Given a path, loads the entity/relation2id files and the embeddings.
    Meant for use on the graphvite pretrained models (.pkl files).

    Parameters
    ----------
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

    if return_human_ents_rels:
        # Exploration: Do all entities and relations of HumanWikidata5M have pre-trained embeddings?
        set_of_graphvite_entities = set(model.graph.id2entity)
        set_of_graphvite_relations = set(model.graph.id2relation)
        with open(os.path.join(BASE_PATH_HOST,
                               'data/interim/KG_and_LM_files/entities_04042022_v1.txt'), 'r') as f:
            HumanWikidata5M_entities = set(f.read().splitlines())
        f.close()
        with open(os.path.join(BASE_PATH_HOST,
                               'data/interim/KG_and_LM_files/relations_04042022_v1.txt'), 'r') as f:
            HumanWikidata5M_relations = set(f.read().splitlines())
        f.close()
        with open(os.path.join(BASE_PATH_HOST,
                               'data/interim/wikidata5m_human_entities_01042022_v2.tsv'), 'r') as f:
            HumanWikidata5M_humans = set(f.read().splitlines())
        f.close()

        # find the entities that are in my human entities, but that do not have a graphvite embedding
        # A.difference(B) = all elements that are in A but not in B
        # 41 QIDs are in my human subset, but not in the pre-trained Graphvite model
        entities_with_embedding = HumanWikidata5M_entities.intersection(set_of_graphvite_entities)
        entities_without_embedding = HumanWikidata5M_entities.difference(set_of_graphvite_entities)
        assert len(HumanWikidata5M_entities) - len(entities_without_embedding) == len(
            entities_with_embedding)
        # check for gender tail value male - Q6581097 and female - Q6581072
        'Q6581097' in entities_without_embedding  # True
        'Q6581072' in entities_without_embedding  # True

        humans_without_embedding = HumanWikidata5M_humans.difference(set_of_graphvite_entities)
        # All relations have a graphvite embedding!
        relations_without_embedding = HumanWikidata5M_relations.difference(
            set_of_graphvite_relations)

        # Create a special set of entities, relations and facts to be used with Graphvite embeddings

        # now filter out all the facts with the missing QIDs
        human_triples_v3 = pd.read_csv(os.path.join(BASE_PATH_HOST,
                                                    'data/interim/wikidata5m_human_facts_with_binary_gender_added_01042022_v3.tsv'),
                                       sep = '\t',
                                       names = ['head_entity', 'relation', 'tail_entity'])

        # This basically corresponds to Step 6 in wikidata5m_preprocessing.py
        entities_without_embedding_list = list(entities_without_embedding)
        human_triples_to_remove = human_triples_v3.query(
            'head_entity == @entities_without_embedding_list | tail_entity == @entities_without_embedding_list')
        # This dataframe has 1,501,977 rows that will be excluded.
        # (Of these, 1,501,938 are gender facts.)

        # The remaining human subset contains 9612820 triples.
        human_triples_graphvite = human_triples_v3.drop(index = human_triples_to_remove.index)

        folder_to_save = os.path.join(BASE_PATH_HOST, 'data/interim/KG_only_files')

        # save to TSV file (184 MB)
        human_triples_graphvite.to_csv(os.path.join(folder_to_save,
                                                    'wikidata5m_human_facts_for_graphvite_models_08042022_v1.tsv'),
                                       sep = '\t', header = False, index = False)

        ### entities.txt: list of all remaining 1731980 entities in the dataset
        set_of_head_entities = set(human_triples_graphvite['head_entity'])
        set_of_tail_entities = set(human_triples_graphvite['tail_entity'])
        set_of_all_remaining_entities = set_of_head_entities.union(set_of_tail_entities)
        with open(os.path.join(folder_to_save, 'entities_for_graphvite_models_08042022_v1.txt'),
                  'w') as f:
            f.writelines(''.join([str(x) + '\n' for x in set_of_all_remaining_entities]))
        f.close()

        ### relations.txt: list of all remaining 291 relations in the dataset (P21 removed)
        set_of_remaining_relations = set(human_triples_graphvite['relation'])
        with open(os.path.join(folder_to_save, 'relations_for_graphvite_models_08042022_v1.txt'),
                  'w') as f:
            f.writelines(''.join([str(x) + '\n' for x in set_of_remaining_relations]))
        f.close()

        print('Wrote all new files to disk.')

    return (entity2id, relation2id), (entity_embeddings, relation_embeddings)


def convert2humanembeddings(embedding_name, entlist, rellist):
    """

    Parameters
    ----------
    embedding_name: String that depicts the model name:

    Returns
    -------
    saves the human embeddings to disk
    """
    model_path = os.path.join(folder_for_loading_and_saving_models,
                              f'{embedding_name}_wikidata5m.pkl')
    (ent2id, rel2id), (entity_embeddings, relation_embeddings) = load_embeddings(model_path)
    # retrieve numeric IDs for all desired entities and relations
    human_ent_ids = [ent2id[i] for i in entlist]
    human_rel_ids = [rel2id[i] for i in rellist]
    # based on the numeric IDs, extract the corresponding embedding vectors
    human_entity_embeddings = entity_embeddings[human_ent_ids]
    human_relation_embeddings = relation_embeddings[human_rel_ids]

    # Store human embeddings in a re-identifiable format
    # Should have the same properties as a pykeen model
    import torch
    pykeen_model = torch.load(os.path.join(BASE_PATH_HOST,
                                           'results/KG_only/TransE_fullW5M_80epochs/checkpoints/checkpoint_TransE_fullW5M_80epochs.pt'),
                              map_location=torch.device('cpu'))


    pickle.dump((human_entity_embeddings, human_relation_embeddings),
                os.path.join(folder_for_loading_and_saving_models,
                             f'{embedding_name}_pretrained_human_W5M.pkl'))


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
folder_for_loading_and_saving_models = os.path.join(BASE_PATH_HOST,
                                                    'trained_models/KG_only/graphvite_pretrained_W5M')

with open(os.path.join(BASE_PATH_HOST, 'data/interim/KG_only_files/entities_for_graphvite_models_08042022_v1.txt'),
          'r') as f:
    graphvite_entities = set(f.read().splitlines())
f.close()
with open(os.path.join(BASE_PATH_HOST, 'data/interim/KG_only_files/relations_for_graphvite_models_08042022_v1.txt'),
          'r') as f:
    graphvite_relations = set(f.read().splitlines())
f.close()

for e in embeddings:
    convert2humanembeddings(e, graphvite_entities, graphvite_relations)
