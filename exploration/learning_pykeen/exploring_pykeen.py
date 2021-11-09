# %% Imports

import pykeen
from pykeen.pipeline import pipeline


# %% Explore the basic training: pipeline

first_results = pipeline(
    dataset = 'Nations',
    model = 'TransE',
    random_seed = 42,
    device = 'cpu'
)



# %% Looking at Wikidata5M

from pykeen.datasets import Wikidata5M


triples_factory_W5M = Wikidata5M()
triples_factory_W5M.factory_dict

# %% Pykeen + GPUS

# TODO try to run a model training on HPIs GPUs