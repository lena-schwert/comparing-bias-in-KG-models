"""
Bias Measurement Classes
Currently implemented : Demographic Parity, Predictive Parity, Translational Likelihood Bias (TLB)
"""
import logging
import time

import torch
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, accuracy_score, recall_score
import pykeen.models

logger = logging.getLogger(__name__)


class Measurement:
    def __init__(self):
        self.name = ""
        self.require_preds_df = True

    def calculate(self, predictions, relation):
        pass

    def get_name(self):
        return self.name


class DemographicParity(Measurement):
    def __init__(self):
        super(DemographicParity, self).__init__()
        self.name = "demographic_parity"

    def calculate_one_relation(self, preds_df, relation):
        """
        Calculate demographic parity for each of the relation tail values
        a.k.a. Group fairness/statistical parity/equal acceptance rate
        A classifier satisfies this definition if subjects in both protected and unprotected groups
        have equal probability of being assigned to the positive predicted class : P(d=1|G=m)=P(d=1|G=f)

        Return:
        demographic_parity_distance: float
        """
        ### prepare data objects needed for the loop that calculates bias
        # extract list of unique true labels from the preds_df, e.g. [0, 1, 2, 3, 4, 5]
        true_target_values = list(set(preds_df.true_tail.values))
        true_target_values = [trgt for trgt in true_target_values if trgt in set(preds_df.pred)]
        attributes = preds_df[relation]
        logger.info(f'{len(attributes[attributes != -1])} people have information on this relation')

        # create list of unique values for current bias relation
        attribute_set = list(set(attributes))

        # remove -1 from the attribute as this indicates a missing value
        if -1 in attribute_set:
            attribute_set.remove(-1)

        # return bias_score = 0 if no person has this attribute
        if len(attribute_set) == 0:
            return 0

        demographic_parity_distance = 0
        for attr in attribute_set:
            # sum demographic_parity_distance over all attributes a, by calculating the difference
            # P[y=target_val|a=relation_val] - p[y=target_val|a!=relation_val]
            # for all true target values
            for target_val in true_target_values:
                # identify all rows where entities are predicted to have current target_value occupation
                # e.g. All women that are predicted to have the occupation politician.
                pred_current_target = (preds_df.pred == target_val)

                # from this, select only the facts/rows where the current attribute value
                # e.g. From the previous vector, only select the rows where the person is female.
                # len(given_a) = sum(attributes == attr) =  number of rows selected
                given_a = pred_current_target[attributes == attr]

                # then select only the facts/rows that are NOT equal to the current attribute value
                # TODO Isn't this a mistake and -1 values should be filtered out here?
                # from predictive parity: given_not_a = preds_df[np.logical_and(preds_df[rel] != attr, preds_df[rel] != -1)]
                given_not_a = pred_current_target[attributes != attr]
                assert len(given_a) + len(given_not_a) == len(pred_current_target), 'Sanity check failed.'

                # Which percentage of people having this attribute are predicted to have this profession?
                # in math this is: P[y = target_val | a = relation_val]
                prob_y_given_a = sum(given_a) / len(given_a)

                # Which percentage of people NOT having this attribute are predicted to have this profession?
                # in math this is: P[y = target_val | a != relation_val]
                prob_y_given_not_a = sum(given_not_a) / len(given_not_a)

                # What is the probability overall to prediction of target profession target_val?
                #sum(pred_current_target) / len(pred_current_target)

                # alternative:
                #given_not_a_without_NA = pred_current_target[np.logical_and(attributes != attr, attributes != -1)]
                #prob_y_given_not_a_without_NA = sum(given_not_a_without_NA) / len(given_not_a_without_NA)
                #demographic_parity_distance_alternative = abs(prob_y_given_a - prob_y_given_not_a_without_NA)

                # Calculate the difference of the probabilities P(y=t|a) and P(y=t|not a),
                # We note that P(y=t|a) + P(y=t|~a) = P(y=t)
                # Therefore |P(y=t|a) - P(y=t|~a)| <= P(y=t)
                # moreover, sum(P(y=t) for all t) = 1
                # So sum(|P(y=t|a) - P(y=t|~a)|  for all t) <= 1
                demographic_parity_distance += abs(prob_y_given_a - prob_y_given_not_a)

        # Normalize the demographic parity distance score, to get a value between 0 and 1
        # TODO Why is this normalized by + 1? Isn't this incorrect and should be omitted?
        demographic_parity_distance = demographic_parity_distance / (len(attribute_set))
        return demographic_parity_distance

    def calculate(self, evaluator, bias_relations):
        """
        Calculate demographic parity distance of possibly biased relations, return a table of demographic parity distances(DPD)
        
        Param:
        =======
        evaluator: bias evaluator
        bias_relations: a list of possibly biased relations to be measured for DPD scores

        Return:
        =======
        dp_df: pandas.DataFrame, a table of DPD scores of input bias_relations
        """
        preds_df = evaluator.predictions
        bias_scores = []
        for r in bias_relations:
            logger.info(f"Calculating bias score for relation: {r}")
            bias_scores.append(self.calculate_one_relation(preds_df, r))
        dp_df = pd.DataFrame({"relations": bias_relations, "bias_scores": bias_scores})
        return dp_df

    # TODO this function is never used anywhere!
    def demographic_parity_for_target_attribute_pair(self, preds_df, relation, attr, target_val):
        attributes = preds_df[relation]
        pred_current_target = (preds_df.pred == target_val)

        given_a = pred_current_target[attributes == attr]
        given_not_a = pred_current_target[attributes != attr]

        prob_y_given_a = sum(given_a) / len(given_a)
        prob_y_given_not_a = sum(given_not_a) / len(given_not_a)
        return abs(prob_y_given_a - prob_y_given_not_a)

    # TODO this function is never used anywhere!
    def demographic_parity_for_target(self, preds_df, relation, target_val):
        attributes = preds_df[relation]
        attribute_set = list(set(attributes))
        pred_current_target = (preds_df.pred == target_val)
        DP = 0
        if -1 in attribute_set:
            attribute_set.remove(-1)

        if len(attribute_set) == 0:
            return 0

        for attr in attribute_set:
            given_a = pred_current_target[attributes == attr]
            given_not_a = pred_current_target[attributes != attr]

            prob_y_given_a = sum(given_a) / len(given_a)
            prob_y_given_not_a = sum(given_not_a) / len(given_not_a)
            DP += abs(prob_y_given_a - prob_y_given_not_a)
        return DP


class PredictiveParity(Measurement):
    def __init__(self):
        super(PredictiveParity, self).__init__()
        self.name = "predictive_parity"

    def calculate_one_relation(self, preds_df, rel):
        """
        Predictive parity (a.k.a. outcome test)
        A classifier satisfies this definition if both protected and unprotected groups
        have equal PPV – the probability of a subject with positive predictive value to
        truly belong to the positive class : P(Y=1|d=1,G=m)=P(Y=1|d=1,G=f)

        Return:
        predictive_parity_distance: float
        """
        attributes = preds_df[rel].values
        attribute_set = list(set(attributes))
        logger.info(f'{len(attributes[attributes != -1])} people have information on this relation')

        if -1 in attribute_set:
            attribute_set.remove(-1)

        if len(attribute_set) <= 1:
            return 0

        predictive_parity_distance = 0
        for attr in attribute_set:
            # sum over all attributes, i.e. tail values for the relation,
            # by calculating the difference
            # E[y=target_val|ytrue=target_val, a=relation_val] - E[y==target_val|ytrue=target_val, a!=relation_val]
            # for all target values
            given_a = preds_df[preds_df[rel] == attr]
            given_not_a = preds_df[np.logical_and(preds_df[rel] != attr, preds_df[rel] != -1)]

            precision_given_a = precision_score(given_a.true_tail, given_a.pred, average = 'micro')
            precision_given_not_a = precision_score(given_not_a.true_tail, given_not_a.pred,
                                                    average = 'micro')
            predictive_parity_distance += abs(precision_given_a - precision_given_not_a)

        predictive_parity_distance = predictive_parity_distance / (len(attribute_set))
        return predictive_parity_distance

    def calculate(self, evaluator, bias_relations):
        """
        Calculate the predictiive parity distance of each possibly biased relation, return a table of predictive parity distances(PPD)
        
        Param:
        =======
        evaluator: bias evaluator
        bias_relations: a list of possibly biased relations to be measured for PPD scores

        Return:
        =======
        dp_df: pandas.DataFrame, a table of PPD scores of input bias_relations
        """
        preds_df = evaluator.predictions
        bias_scores = []
        for r in bias_relations:
            logger.info(f"{r}")
            bias_scores.append(self.calculate_one_relation(preds_df, r))
        pp_df = pd.DataFrame({"relations": bias_relations, "bias_scores": bias_scores})
        return pp_df

