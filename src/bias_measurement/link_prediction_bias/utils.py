import logging
from datetime import datetime
import os
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def get_classifier(dataset, target_relation, num_classes, batch_size, embedding_model_path,
                   load_trained_classifier: bool, trained_classifier_filename: str,
                   classifier_type = 'mlp', **model_kwargs):
    """
    Return a classifier that will classify the tails for the target relations
    Currently only MLP classifier is implemented, but can look into others
    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    target_relation: str,
    num_classes: int, number of classification classes
    embedding_model_path: str, path to trained embedding model using pykeen library
    """

    if classifier_type == 'mlp':
        from src.bias_measurement.link_prediction_bias.classifier import TargetRelationClassifier
        return TargetRelationClassifier(dataset = dataset,
                                        embedding_model_path = embedding_model_path,
                                        target_relation = target_relation,
                                        num_classes = num_classes,
                                        batch_size = batch_size,
                                        load_trained_classifier = load_trained_classifier,
                                        trained_classifier_filename = trained_classifier_filename,
                                        **model_kwargs)
    elif classifier_type == 'rf':
        if load_trained_classifier:
            # TODO implement loading of existing classifier for this model class
            raise NotImplementedError
        from classifier import RFRelationClassifier
        return RFRelationClassifier(dataset = dataset,
                                    embedding_model_path = embedding_model_path,
                                    target_relation = target_relation,
                                    num_classes = num_classes,
                                    batch_size = batch_size,
                                    max_depth = 3,
                                    class_weight = 'balanced',
                                    max_features = 'auto',
                                    **model_kwargs)


def get_sensitive_and_target_relations(dataset_name):
    if dataset_name.lower() == "wikidata5m":
        target_relations = ['P106']  # occupation
        sensitive_relations = ['P21']  # sex or gender
        # The k most common relations in Wikidata5M (old version)
        # P27 - country of citizenship                  #######################################
        # P54 - member of sports team                   #######################################
        # P735 - given name                             #######################################
        # P19 - place of birth                          #######################################
        # P69 - educated at                             #######################################
        # P641 -  sport                                   #######################################
        # P20 - place of death                          #######################################
        # P1412 - languages spoken, written or signed   #######################################
        # P1344 - participant in                        #######################################
        # P413 - position played on team/specialty      #######################################
        # P166 - award received                         #######################################
    else:
        raise NotImplementedError('Other datasets than Wikidata5M are currently not implemented.')

    # This is required such that the relations can be looped through meaningfully
    # relevant for predict_tails.add_relation_values(), Measurement.py
    assert type(sensitive_relations) == list
    assert type(target_relations) == list

    all([type(item) == str for item in sensitive_relations])
    all([type(item) == str for item in target_relations])

    return sensitive_relations, target_relations


def save_result(result, dataset, args):
    """
    Save dataset summary, and output from Evaluator

    result: dict, bias evaluation result
    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    args: arguments passed when running main program
    """
    if args.embedding_path:
        embedding = os.path.splitext(os.path.split(args.embedding_path)[-1])[0]
    else:
        embedding = args.embedding_name
    date = datetime.now().strftime("%d.%m.%Y_%H:%M")
    dir = "./results_" + date
    if not os.path.exists(dir):
        os.makedirs(dir)  # Save Dataset Summary
    with open(os.path.join(dir, args.dataset_name + ".txt"), 'w') as f:  # save dataset summary
        f.writelines(dataset.summary_str())  # TODO: save embedding training configuration?
    for k in result.keys():
        #measure_dir = os.path.join(dir, k)
        #os.mkdir(measure_dir)
        if isinstance(result[k], pd.DataFrame):
            save_path = os.path.join(dir, "{}.csv".format(k))
            logger.info("Save to {}".format(save_path))
            result[k].to_csv(save_path)
        elif isinstance(result[k], dict):
            for rel in result[k].keys():
                df = pd.DataFrame(result[k][rel])
                rel = rel.split('/')[-1] if args.dataset_name == 'fb15k237' else rel
                save_path = os.path.join(dir, "{}_{}.csv".format(k, rel))
                logger.info("Save to {}".format(save_path))
                df.to_csv(save_path)


def remove_infreq_attributes(attr_counts, key, threshold = 10, nan_val = -1):
    if attr_counts[key] <= threshold:
        return nan_val
    return key


def requires_preds_df(bias_measures):
    """
    :param bias_measures: a list of bias metrics
    :return: bool, True if we need a preds dataframe and False if not
    """
    require_preds_df = False
    for m in bias_measures:
        if m.require_preds_df:
            require_preds_df = True
            break
    return require_preds_df
