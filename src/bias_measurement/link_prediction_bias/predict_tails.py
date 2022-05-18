""""
Create a data-frame with tail predictions
This data-frame can later be used to measure bias
"""
from pykeen.datasets import FB15k237
import pandas as pd
import numpy as np
import os
import torch
from collections import Counter
from datetime import datetime
import logging

#### Internal Imports
from src.bias_measurement.link_prediction_bias.utils import get_classifier, remove_infreq_attributes
from src.bias_measurement.link_prediction_bias.BiasEvaluator import BiasEvaluator

# makes it compatible with logging coming from other sources
logger = logging.getLogger(__name__)


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

    # access the test triples of the dataset which have the bias relations (as string ID triples)
    bias_relations_triplets_mask = dataset.testing.get_mask_for_relations(bias_relations)
    bias_relations_triplets = dataset.testing.triples[bias_relations_triplets_mask]
    # only select the bias_relation facts for which the human head entity is part of preds_df['entity']
    #bias_relations_triplets = [tr for tr in bias_relations_triplets if dataset.entity_to_id[tr[0]] in preds_df['head_entity'].values]
    entity_to_tail = {}
    # for each bias relation, create a dict entry, where:
    # key = bias relation string, value = empty dict
    # e.g. {'P21': {}}
    for bias_rel in bias_relations:
        entity_to_tail[bias_rel] = {}
    # for each bias relation triple, add the numeric head and tail ID to entity_to_tail
    # e.g. {'P21': {379209: 1370033, 763948: 1370033}}
    for head, rel, tail in bias_relations_triplets:
        # retrieve numeric ID for head and tail entity
        head_id = dataset.entity_to_id.get(head)
        tail_id = dataset.entity_to_id.get(tail)
        # create a dict entry, where key = num ID head, value = num ID tail
        entity_to_tail[rel][head_id] = tail_id
    # for each bias relation, create a column of numeric ID tail values for the corresponding human head entity
    for bias_rel in bias_relations:
        # for each human head_entity in preds_df, retrieve the numeric ID for the corresponding tail
        # value will be -1 if this fact does not exist in the test set
        # TODO maybe retrieve the sensitive relation facts from the entire dataset instead?
        preds_df[bias_rel] = [get_tail(bias_rel, head_entity) for head_entity in preds_df['head_entity'].values]
        # count the occurrence
        attr_counts = Counter(preds_df[bias_rel])
        # IMPORTANT: set a threshold for removing facts that are considered too rare
        #preds_df[rel] = preds_df[rel].apply(lambda x: remove_infreq_attributes(attr_counts, x))
    return preds_df


def predict_relation_tails(dataset, trained_classifier, target_test_triplets):
    """
    predict the tail t for (h,r,t)
    for each head entity h in the dataset
    return a dataframe with the predictions - each row is an entity

    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237, or wikidata 5m
    trained_classifier: a classifier (mlp or random forest) trained to classify head
        entities into the target relation classes
    target_test_triplets: triples to classify (i.e. predict tails for)
    """
    # create a dataframe from the test triples using the dataset IDs as strings
    preds_df = pd.DataFrame(
        {'head_entity': target_test_triplets[:, 0], 'relation': target_test_triplets[:, 1],
         'true_tail': target_test_triplets[:, 2], })
    # map the string IDs to pykeens numeric IDs
    preds_df['head_entity'] = preds_df['head_entity'].apply(lambda head: dataset.entity_to_id.get(head))

    # doublecheck whether created preds_df does contain any bias facts at all
    assert preds_df.empty is False

    preds_df.to_csv('all_target_test_triples.tsv', sep = '\t')

    # prepare everything for tail prediction using trained classifier
    heads = torch.Tensor(preds_df['head_entity'].values)
    heads = heads.long()
    target_relation = list(preds_df.relation.unique())
    preds_df['pred'] = trained_classifier.predict_tails(heads = heads, relation = target_relation)

    # map true tails from strings to pykeen IDs
    preds_df['true_tail'] = preds_df['true_tail'].apply(
        lambda tail_entity: dataset.entity_to_id[tail_entity])
    # map pykeen IDs to classification targets
    preds_df['true_tail'] = preds_df['true_tail'].apply(
        lambda tail_entity: trained_classifier.target2label(tail_entity))

    return preds_df


def get_preds_df(dataset, classifier_args, model_args, target_relation, bias_relations,
                 preds_df_path: str = None):
    """
    Get predictions dataframe used in parity distance calculation

    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    classifier_args: dict, parameters passed to train classifier
    model_args: dict,
    target_relation: str,
    bias_relations: list of str,
    preds_df_path: str, path to predictions dataframe, default to None
    """
    if preds_df_path is not None and os.path.exists(preds_df_path):
        # If a dataframe already exists, read it instead of creating it
        preds_df = pd.read_csv(preds_df_path)
        del preds_df['Unnamed: 0']
        logger.info(f"Load predictions dataframe from: {preds_df_path}")
        return preds_df

    classifier = get_classifier(dataset = dataset, target_relation = target_relation,
                                num_classes = classifier_args["num_classes"],
                                batch_size = classifier_args["batch_size"],
                                embedding_model_path = model_args['embedding_model_path'],
                                classifier_type = classifier_args["type"],
                                load_trained_classifier = classifier_args['load_trained_classifier'],
                                trained_classifier_filename = classifier_args['trained_classifier_filename'])

    # train classifier + save it
    if not classifier_args['load_trained_classifier']:
        if classifier_args["type"] == "mlp":
            classifier.train(classifier_args['epochs'])
            #torch.save(classifier, 'trained_model_TargetRelationClassifier.pt')
            # model is saved internally by pytorch ignite after last epoch
        elif classifier_args["type"] == "rf":
            classifier.train()  # TODO save the trained classifier

    # from all test triples, only select those where the relation is the target relation, here: occupation
    target_test_triplets_mask = dataset.testing.get_mask_for_relations(target_relation)
    target_test_triplets = dataset.testing.triples[target_test_triplets_mask]

    # get predictions dataframe
    preds_df = predict_relation_tails(dataset, classifier, target_test_triplets)
    preds_df = add_relation_values(dataset, preds_df, bias_relations)

    return preds_df


# TODO: Pass less default param, maybe add kwargs
def eval_bias(evaluator, classifier_args, model_args, bias_relations = None, bias_measures = None,
              preds_df_path = None, ):
    """
    Creates a predictions dataframe & evaluates bias on it
    evaluator: instance of Evaluator(see BiasEvaluator.py),
    classifier_args: dict,
    model_args: dict,
    bias_relations: list of str,
    bias_measures: list of instances of Measurement,
    preds_df_path: str, path to predictions
    """
    from utils import requires_preds_df
    target_relation = evaluator.target_relation
    dataset = evaluator.dataset
    if requires_preds_df(bias_measures):
        preds_df = get_preds_df(dataset = dataset, classifier_args = classifier_args,
                                model_args = model_args, target_relation = target_relation,
                                bias_relations = bias_relations, preds_df_path = preds_df_path, )
        logger.info("Got predictions dataframe")
        evaluator.set_predictions_df(preds_df)
    eval_bias = evaluator.evaluate_bias(bias_relations, bias_measures)
    return eval_bias
