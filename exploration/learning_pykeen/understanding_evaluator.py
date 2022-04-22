from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from pykeen.models import TransE, ComplEx, DistMult, RotatE


first_results = pipeline(
    dataset = 'fb15k237',
    model = 'TransE',
    model_kwargs = dict(
        embedding_dim = 100,
    ),
    random_seed = 42,
    device = 'cpu',
    filter_validation_when_testing = True,   # manually enable/confirm Bordes 2013 filtered setting
    training_kwargs = dict(num_epochs = 1)
)

first_results.metric_results

# %% How does pykeen use its RankBased Evaluator?
