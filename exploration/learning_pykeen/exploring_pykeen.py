# %% Imports
import os
import pykeen
from pykeen.pipeline import pipeline
import pandas as pd


base_path_of_repo = os.path.join(os.getcwd(), 'exploration/learning_pykeen')

# %% Explore the basic training: pipeline


first_results = pipeline(
    dataset = 'Nations',
    model = 'TransE',
    model_kwargs = dict(
        embedding_dim = 100,
    ),
    random_seed = 42,
    device = 'cpu',
    filter_validation_when_testing = True   # manually enable/confirm Bordes 2013 filtered setting
)

# save everything to disk
first_results.save_to_directory(os.path.join(base_path_of_repo, 'first_results'))

results_df = first_results.metric_results.to_df()
# only keep side = both, type = realistic as it is recommended
results_df = results_df[(results_df.Side == 'both') & (results_df.Type == 'realistic')]


# %% Looking at Wikidata5M + simple training

from pykeen.datasets import Wikidata5M

### type: pykeen.datasets.wikidata5m.Wikidata5M
W5M_internal = Wikidata5M()

pipeline_W5M_internal = pipeline(
    dataset = 'Wikidata5M',
    model = 'TransE',
    #negative_sampler = 'hm',
    epochs = 10,
    random_seed = 42,
    device = 'cpu'
)

# save pipeline result to disk

from pykeen.datasets.analysis import *

get_relation_count_df(W5M_internal)
get_entity_count_df(W5M_internal)
# get counts for combination of entites and relations
pykeen_rel_ent_cooccur_W5M = get_entity_relation_co_occurrence_df(W5M_internal)
# analyze relation types: symmetry, anti-symmetry, inversion, composition
pykeen_relation_pattern_types_w5M = get_relation_pattern_types_df(W5M_internal)


# save the large files to disk
pykeen_rel_ent_cooccur_W5M.to_pickle(os.path.join(base_path_of_repo, 'pykeen_rel_ent_co_occur_W5M_internal.pkl'))
pykeen_relation_pattern_types_w5M.to_pickle(os.path.join(base_path_of_repo, 'pykeen_relation_pattern_types_W5M_internal.pkl'))


# %% explore TriplesFactory properties

# load Wikidata5M indirectly using file
from pykeen.triples import TriplesFactory
path_to_file = os.path.join(base_path_of_repo, 'data/SOTA_datasets_raw_downloads/Wikidata5M/wikidata5m_all_triplets.txt')
### type: pykeen.triples.triples_factory.TriplesFactory
triples_factory_W5M = TriplesFactory.from_path(path_to_file)

### explore Wikidata5M using pykeen
triples_factory_W5M.num_entities  # about 5 million
triples_factory_W5M.num_relations  # 828

most_frequent_rel_100 = triples_factory_W5M.get_most_frequent_relations(100)
# retrieve Wikidata labels for this set of IDs
most_frequent_PIDs_100 = [triples_factory_W5M.relation_id_to_label.get(key) for key in list(most_frequent_rel_100)]

# plot word cloud of relations
word_cloud_ent_top100 = triples_factory_W5M.entity_word_cloud(100)
word_cloud_rel_top100 = triples_factory_W5M.relation_word_cloud(100)

# save word clouds as HTML files
# with open(os.path.join(base_path_of_repo, 'word_cloud_ent_top100.html'), 'w') as file:
#     file.write(word_cloud_ent_top100.data)
# with open(os.path.join(base_path_of_repo, 'word_cloud_rel_top100.html'), 'w') as file:
#     file.write(word_cloud_rel_top100.data)

training, validation, testing = triples_factory_W5M.split([0.8, 0.1, 0.1],
                                                          random_state = 42)

TransE_W5M = pipeline(
    training = training,
    validation = validation,
    testing = testing,

)


# creating triples factory from Wikidata5M all triplets on disk
from pykeen.triples import TriplesFactory


# %% Pykeen + GPUS

# TODO try to run a model training on HPIs GPUs


