







def create_preds_df():
    """
    1. This uses the predict_tails.get_preds_df() function from Keidar.
    2. This function then calls utils.get_classifier().
    3. classifier.train() then trains a occupation classifier based on the embeddings
       of the link prediction model.
    4. predict_tails.predict_relation_tails() creates the first 4 columns of the preds_df:
       entity, relation, true tail, preds (by the classifier)
       This shows true labels for occupation and what the classifier predicted.
    5. predict_tails.add_relation_values() gets the sensitive attribute values for each person
       in preds_df from the original dataset, e.g. who is female/male/value missing.
    6. This dataframe is then stored as CSV, one preds_df per model.

    Arguments needed for this are:
        dataset = a dataset object, originally inherited from pykeen.PathDataset(LazyDataset(Dataset))
        classifier_args = epochs, batch size, type (e.g. MLP), number of classes
        model_args = the path to a trained link prediction model
        target_relation = an identifier of the target relation, will be 'P106' (occupation) for W5M
        bias_relations = list of identifier for relations that bias should be calculated for
        (originally in utils.suggest_relations: ['P27', 'P735', 'P19', 'P54', 'P69', 'P641', 'P20', 'P1344', 'P1412', 'P413'])


    Returns
    -------

    """

    pass


de

