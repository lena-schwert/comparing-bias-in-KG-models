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
    bias_relations_triplets_mask = dataset.testing.get_mask_for_relations([bias_relations])
    bias_relations_triplets = dataset.testing.triples[bias_relations_triplets_mask]
    # only select the bias_relation facts for which the human head entity is part of preds_df['entity']
    #bias_relations_triplets = [tr for tr in bias_relations_triplets if dataset.entity_to_id[tr[0]] in preds_df['head_entity'].values]
    entity_to_tail = {}
    # for each bias relation, create a dict entry, where:
    # key = bias relation string, value = empty dict
    # e.g. {'P21': {}}
    for bias_rel in [bias_relations]:
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
    for bias_rel in [bias_relations]:
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

    # prepare everything for tail prediction using trained classifier
    heads = torch.Tensor(preds_df['head_entity'].values)
    heads = heads.long()
    target_relation = preds_df.relation.loc[0]
    preds_df['pred'] = trained_classifier.predict_tails(heads = heads, relation = target_relation)

    # map true tails from strings to pykeen IDs
    preds_df['true_tail'] = preds_df['true_tail'].apply(
        lambda tail_entity: dataset.entity_to_id[tail_entity])
    # map pykeen IDs to classification targets
    preds_df['true_tail'] = preds_df['true_tail'].apply(
        lambda tail_entity: trained_classifier.target2label(tail_entity))

    return preds_df


def get_preds_df(dataset, classifier_args, model_args, target_relation, bias_relations,
                 preds_df_path = None):
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
                                classifier_type = classifier_args["type"])

    # train classifier + save it
    if classifier_args["type"] == "mlp":
        classifier.train(classifier_args[
                             'epochs'])  # model is saved internally by pytorch ignite after last epoch
    elif classifier_args["type"] == "rf":
        classifier.train()  # TODO save the trained classifier

    # from all test triples, only select those where the relation is the target relation, here: occupation
    target_test_triplets_mask = dataset.testing.get_mask_for_relations([target_relation])
    target_test_triplets = dataset.testing.triples[target_test_triplets_mask]

    # get predictions dataframe
    preds_df = predict_relation_tails(dataset, classifier, target_test_triplets)
    preds_df = add_relation_values(dataset, preds_df, bias_relations)

    # save predictions if a path is specified
    if preds_df_path is not None:
        # preds_df.to_csv(preds_df_path)
        preds_df.to_csv('preds_df_transe_created_by_lena.csv')
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


if __name__ == '__main__':
    # IMPORTANT calculation of bias scores based on preds_df starts here

    START_TIME = datetime.now()

    import argparse
    from classifier import RFRelationClassifier
    from Measurement import DemographicParity, PredictiveParity, TranslationalLikelihood
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    from visualization import preds_histogram
    from collections import Counter

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'fb15k237',
                        help = "Dataset name, must be one of fb15k237. Default to fb15k237.")
    parser.add_argument('--embedding', type = str, default = 'trans_e', help = "Embedding name, must be one of complex, conv_e, distmult, rotate, trans_d, trans_e. \
                                Default to trans_e")
    parser.add_argument('--embedding_path', type = str, help = "Specify a full path to your trained embedding model. It will override default path \
                              inferred by dataset and embedding")
    parser.add_argument('--predictions_path', type = str, help = 'path to predictions used in parity distance, specifying \
                               it will override internal inferred path')
    parser.add_argument('--epochs', type = int,
                        help = "Number of training epochs of link prediction classifier (used for DP & PP), default to 100",
                        default = 100)
    args = parser.parse_args()

    # Trained Embedding Model Path
    # TODO change it back
    MODEL_PATH = '/home/lena/git/master_thesis_bias_in_NLP/results/TransE_fullW5M_80epochs/trained_model.pkl'

    # embedding_model_path_suffix = "replicates/replicate-00000/trained_model.pkl"
    # MODEL_PATH = os.path.join(LOCAL_PATH_TO_EMBEDDING, args.dataset, args.embedding, embedding_model_path_suffix)
    # if args.embedding_path:
    #     MODEL_PATH = args.embedding_path  # override default if specifying a full path
    logger.info("Load embedding model from: {}".format(MODEL_PATH))

    measures = [DemographicParity(), PredictiveParity()]

    # Init dataset and relations of interest
    # if args.data
    dataset = FB15k237()
    target_relation, bias_relations = ('P106', 'P21')

    # Init embed model and classifier parameter
    model_args = {'embedding_model_path': MODEL_PATH}

    classifier_args = {'epochs': args.epochs, "batch_size": 256, "type": 'rf', 'num_classes': 6}

    # TODO load an existing preds_df (unnecesary, done in loop)
    # PREDS_DF_PATH = '/home/lena/git/master_thesis_bias_in_NLP/code_from_other_papers/Keidar_automatic_bias_detec/preds_dfs/preds_df_transe.csv'
    #
    # preds_df = get_preds_df(dataset, classifier_args, model_args, target_relation, bias_relations,
    #                         preds_df_path = PREDS_DF_PATH)

    # Specify your local file paths here
    # file_names = ['/path/to/fb15k237/distmult/replicates/replicate-00000/trained_model.pkl',
    #               '/path/to/fb15k237/trans_e/replicates/replicate-00000/trained_model.pkl',
    #               '/path/to/fb15k237/conve/replicates/replicate-00000/trained_model.pkl',
    #               '/path/to/fb15k237/rotate/replicates/replicate-00000/trained_model.pkl',
    #               '/path/to/fb15k237/complex/replicates/replicate-00000/trained_model.pkl']

    model_names = ['transe', 'distmult', 'conve', 'rotate', 'complex']

    # TODO in this loop, the PPD and DPD bias scores are calculated for each KGE model!
    for model_name in model_names:
        preds_df = pd.read_csv(f'./preds_dfs/preds_df_' + model_name + '.csv')

        evaluator = BiasEvaluator(dataset, measures)
        # pass the dataframe to self.predictions such that it has access to classification results
        evaluator.set_predictions_df(preds_df)
        bias_eval = evaluator.evaluate_bias(bias_relations = bias_relations,
                                            bias_measures = measures)

        d_parity, p_parity = bias_eval['demographic_parity'], bias_eval['predictive_parity']

        # save results of bias calculation here for each of the two bias criteria
        d_parity.to_csv(f'./preds_dfs/DPD_' + model_name + '_Lena' + '.csv')
        p_parity.to_csv(f'./preds_dfs/PPD_' + model_name + '_Lena' + '.csv')

        acc = accuracy_score(preds_df.pred, preds_df.true_tail)
        bacc = balanced_accuracy_score(y_pred = preds_df.pred, y_true = preds_df.true_tail)
        logger.info(acc)
        logger.info(bacc)

    END_TIME = datetime.now()
    logger.info('Finished calculating bias scores for all models.')
    logger.info(f'Start time: {START_TIME.strftime("%d.%m.%Y %H:%M")}')
    logger.info(f'End time: {END_TIME.strftime("%d.%m.%Y %H:%M")}')
    logger.info(f'Calculation took {END_TIME - START_TIME}')

# %% This code part manually calculates bias for FB15k237

# fname = "/Users/alacrity/Documents/uni/Fairness/trained_model.pkl"
# # Trained Model Path
# # fname = '/local/scratch/kge_fairness/models/fb15k237/transe_openkeparams_alpha1/replicates/replicate-00000/trained_model.pkl'
# dataset = FB15k237()
# dataset_name = 'fb15k237'
# # GENDER_RELATION = '/people/person/gender'
# # PROFESSION_RELATION = '/people/person/profession'
#
# target_relation, bias_relations = suggest_relations(dataset_name)
# num_classes = 6
#
# rf = RFRelationClassifier(dataset = dataset, embedding_model_path = fname,
#     target_relation = target_relation, num_classes = num_classes, batch_size = 500,
#     class_weight = 'balanced', max_depth = 6, random_state = 111, n_estimators = 100, )
#
# rf.train()
#
# target_test_triplets = dataset.testing.get_triples_for_relations([target_relation])
# preds_df = pd.DataFrame(
#     {'entity': target_test_triplets[:, 0], 'relation': target_test_triplets[:, 1],
#      'true_tail': target_test_triplets[:, 2], })
# target_relation = preds_df.relation.loc[0]
#
# preds_df = predict_relation_tails(dataset, rf, target_test_triplets)
# preds_df = add_relation_values(dataset, preds_df, bias_relations)
#
# random_preds = [np.random.randint(num_classes) for __ in preds_df.pred]
# print("classification accuracy for random labels",
#       accuracy_score(preds_df.true_tail, random_preds))
# print("balanced classification accuracy for random labels",
#       balanced_accuracy_score(preds_df.true_tail, random_preds))
#
# print("classification accuracy for rf model", accuracy_score(preds_df.true_tail, preds_df.pred))
# print("balanced classification accuracy for rf model",
#       balanced_accuracy_score(preds_df.true_tail, preds_df.pred))
# preds_histogram(preds_df)
#
# measures = [DemographicParity(), PredictiveParity()]
#
# evaluator = BiasEvaluator(dataset, measures)
# evaluator.set_predictions_df(preds_df)
# bias_eval = evaluator.evaluate_bias(bias_relations = bias_relations, bias_measures = measures)
# d_parity, p_parity = bias_eval['demographic_parity'], bias_eval['predictive_parity']
