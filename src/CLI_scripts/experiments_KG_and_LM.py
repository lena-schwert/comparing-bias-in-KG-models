# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPORTANT: This code is for the most part taken from: https://github.com/yao8839836/kg-bert


# Imports

import argparse
import csv
import logging
import os
import random
import shutil
# in-built modules
import socket
import sys
import time
from datetime import datetime
import copy

# installed modules
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm, trange
from transformers import AdamW, \
    get_linear_schedule_with_warmup  # instead of BertAdam, WarmupLinearSchedule
# imports for BERT
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.file_utils import TRANSFORMERS_CACHE

# imports from my own code
from src.utils import set_base_path_based_on_host, initialize_my_logger, \
    improve_pandas_viewing_options, save_argparse_obj_to_disk

BASE_PATH_HOST = set_base_path_based_on_host()
improve_pandas_viewing_options()

# makes it compatible with logging coming from other sources
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b = None, text_c = None, label = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data. Used for storing results of preparing a single triple
    as tokenized text input for BERT. This class is used by the function convert_examples_to_features()."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar = None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding = "utf-8") as f:
            reader = csv.reader(f, delimiter = "\t", quotechar = quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    # original code: line = list(unicode(cell, 'utf-8') for cell in line)
                    raise SystemError(
                        'sys.version_info[0] == 2, I do not know what that means! Check code')
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(  # set type "train" here for the _create_examples function
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev",
                                     data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test",
                                     data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    # TODO labels [0,1] are hardcoded
    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        logging.debug('Reading train.tsv...')
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        logging.debug('Reading dev.tsv...')
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        logging.debug('Reading test.tsv...')
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        # access to file entity2text.txt
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                # Why is this check necessary?
                if len(temp) == 2:
                    # select the second entry = entity as text
                    end = temp[1]  # .find(',')
                    ent2text[temp[0]] = temp[1]  # [:end]

        # IMPORTANT: I disabled the use of entity2textlong.txt
        # original KG-BERT: only execute this code for FB15K and FB15K-237 datasets
        if data_dir.find("FB15") != -1:
            logger.info(
                'Using longer entity descriptions (entity2textlong.txt) for Freebase datasets.')
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                # first_sent_end_position = temp[1].find(".")
                ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]

        # access list of entities from ent2text
        entities = list(ent2text.keys())

        # obtain the relations from the respective file
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                # make sure to not overwrite anything, enter temp to rel2text
                # key = entity,
                rel2text[temp[0]] = temp[1]

                # Create a set of all the triples in `lines` after concatenating them with `\t` character
        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []

        logging.info('Creating InputExamples, i.e. tokens from entity/relation IDs... ')
        # TODO enable tqdm again!
        # for (i, line) in enumerate(tqdm(lines)):
        for (i, line) in enumerate(lines):

            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":

                # label
                label = "1"

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                self.labels.add(label)
                examples.append(
                    InputExample(guid = guid, text_a = text_a, text_b = text_b, text_c = text_c,
                                 label = label))

            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                # add triple to examples with label 1 = True
                examples.append(
                    InputExample(guid = guid, text_a = text_a, text_b = text_b, text_c = text_c,
                                 label = "1"))

                # generate NEGATIVE/CORRUPT triples, 5 per positive sample!
                # replace either the head or the tail, decide randomly!
                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    # TODO Might it be better move the random sampling inside the loop?
                    # TODO change this baaack
                    # TODO should be range(5)
                    for j in range(1):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            # TODO What if negative triple is in the dev or test set?
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_head_text = ent2text[tmp_head]  # cast corrupt entity to text
                        # append corrupt/negative example to list of InputExample objects with label 0 = False
                        examples.append(
                            InputExample(guid = guid, text_a = tmp_head_text, text_b = text_b,
                                         text_c = text_c, label = "0"))
                else:
                    # corrupting tail
                    tmp_tail = ''
                    for j in range(1):
                        # use while loop to make sure that the negative triple does not exist yet
                        while True:
                            tmp_ent_list = set(entities)  # cast list to set
                            tmp_ent_list.remove(line[2])  # remove tail entity from the set
                            tmp_ent_list = list(tmp_ent_list)  # cast set back to list
                            tmp_tail = random.choice(
                                tmp_ent_list)  # choose a random entity from list
                            # create new TSV-separated triple using new tail entity
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = ent2text[
                            tmp_tail]  # retrieve text version of the corrupt entity
                        # append corrupt/negative example to list of InputExample objects with label 0 = False
                        examples.append(InputExample(guid = guid, text_a = text_a, text_b = text_b,
                                                     text_c = tmp_tail_text, label = "0"))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    logging.info(
        'Creating InputFeatures, i.e. numeric vectors from the previously created InputExample... ')
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    # TODO enable tqdm again!
    # for (ex_index, example) in enumerate(tqdm(examples)):
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # TODO remove this when no longer needed, this is a lot!
        # logging.debug(f'Current example: {example.guid}')

        # tokenize the head entity, outputs a list of strings
        tokens_a = tokenizer.tokenize(example.text_a)
        #logger.debug(f'And the token IDs: {tokenizer.convert_tokens_to_ids(tokens_a)}')
        #logger.debug(f'Tokens of the head entity: {tokens_a}')

        # set tokens to None because they will be changed?
        tokens_b = None
        tokens_c = None

        # this is only executed if both objects are not None
        if example.text_b and example.text_c:
            # tokenize relation and tail entity with the tokenizer
            tokens_b = tokenizer.tokenize(example.text_b)
            #logger.debug(f'Tokens of the relation: {tokens_b}')
            #logger.debug(f'And the token IDs: {tokenizer.convert_tokens_to_ids(tokens_b)}')
            tokens_c = tokenizer.tokenize(example.text_c)
            #logger.debug(f'Tokens of the tail entity: {tokens_c}')
            #logger.debug(f'And the token IDs: {tokenizer.convert_tokens_to_ids(tokens_c)}')
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)  # use for relation prediction only

            # IMPORTANT shorten the whole triple token sequence if necessary to comply with max sequence length
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4" with max_seq_length - 4
            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # IMPORTANT This concept corresponds to segment embeddings described in the paper
        # (head + tail entity have the same, relation has a different one)
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        #  executed for link prediction + triple classification, as variables are filled with strings
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)

        # look up each token in the vocabulary of the tokenizer (the IDs are fixed)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #logger.debug(f'Input IDs created by tokenizer: {input_ids}')

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(
            input_ids) == max_seq_length, "Input ID list must have same length as max sequence length."
        assert len(
            input_mask) == max_seq_length, "Input mask list must have same length as max sequence length."
        assert len(
            segment_ids) == max_seq_length, "Segment ID list must have same length as max sequence length."

        # store the label of the example as an int
        label_id = label_map[example.label]

        # print information about the first 3 examples of the dataset
        if ex_index < 1 and print_info:
            logging.info(f"*** Example {ex_index + 1}***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids = input_ids, input_mask = input_mask, segment_ids = segment_ids,
                          label_id = label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()


def compute_metrics(predictions, true_labels):
    accuracy = Accuracy()

    accuracy = accuracy(predictions, true_labels)

    if args.fp16:
        raise NotImplementedError

    return accuracy

    # from sklearn.metrics import accuracy_score, precision_recall_fscore_support  #  #  # labels = pred.label_ids  # preds = pred.predictions.argmax(-1)  # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')  # acc = accuracy_score(labels, preds)  # return {  #     'accuracy': acc,  #     'f1': f1,  #     'precision': precision,  #     'recall': recall  # }


def train_and_validate():
    START_TRAINING = time.perf_counter()
    #############------------- LOAD MODEL ---------------#################

    ### load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = args.do_lower_case)

    # Prepare model, download BERT for sequence classification
    # create cache folder so the model is only downloaded when the script is run for the first time on the machine
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(TRANSFORMERS_CACHE),
                                                                   'distributed_{}'.format(
                                                                       args.local_rank))

    # parameters will be loaded as float 32
    # loads pretrained BERT model in model.eval() mode
    model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir = cache_dir,
                                                          num_labels = NUM_LABELS)

    # if specified, cast the model to float 16 datatype
    if args.fp16:
        model.half()

    # send the model to device (cpu or gpu)
    model.to(device)

    # if specified, enable distributed training on GPUs using torch.nn.DataParallel(model)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)  # model = torch.nn.parallel.data_parallel(model)

    #############------------- PREPARE TRAINING DATA ---------------#################

    # use gradient accumulation step to set training batch size
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # divide batch size by gradient accumulation steps
    # this is not relevant for the results, usually = 1!
    # // ensures that the result is an integer, not a float!
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # get training examples + create NEGATIVE/CORRUPT triples for each of it
    logger.info('Creating examples with negative sampling from the training split of the dataset.')
    train_examples = processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length,
                                                  tokenizer)
    logger.info('################# TRAINING dataset #################')
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)

    # create 4 separate variables out of train_features
    all_input_ids_train = torch.tensor([f.input_ids for f in train_features], dtype = torch.long)
    all_input_mask_train = torch.tensor([f.input_mask for f in train_features], dtype = torch.long)
    all_segment_ids_train = torch.tensor([f.segment_ids for f in train_features],
                                         dtype = torch.long)
    all_label_ids_train = torch.tensor([f.label_id for f in train_features], dtype = torch.long)

    # wrap all tensors into a torch.TensorDataset
    train_data = TensorDataset(all_input_ids_train, all_input_mask_train, all_segment_ids_train,
                               all_label_ids_train)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)  # default: replacement = False
    else:
        train_sampler = DistributedSampler(train_data)

    # Pytorch data_loader, not shuffled (RandomSampler does that)
    train_dataloader = DataLoader(train_data, sampler = train_sampler,
                                  batch_size = args.train_batch_size)

    #############------------- CONFIGURE OPTIMIZER ---------------#################

    ### Prepare optimizer: account for fp16
    param_optimizer = list(model.named_parameters())

    # IMPORTANT: specify layers where no weight decay should be done
    # (it does not make sense to decay the bias terms)
    # taken from the original bert paper: https://github.com/google-research/bert/blob/master/optimization.py#L65
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # set weight decay = 0 for selected layers
    # model parameters are split into two groups: with and without weight decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    # calculate total number of training steps (needed for learning rate scheduler)
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # use other optimizers if using float16 data type
    if args.fp16:
        raise NotImplementedError(
            'Using fp16, doublecheck all settings!')  # logger.debug('Setting optimizer for float 16 model.')  # try:  #     from apex.optimizers import FP16_Optimizer  #     from apex.optimizers import FusedAdam  # except ImportError:  #     raise ImportError(  #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")  #  # optimizer = FusedAdam(optimizer_grouped_parameters, lr = args.learning_rate,  #                       bias_correction = False, max_grad_norm = 1.0)  # if args.loss_scale == 0:  #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale = True)  # else:  #     optimizer = FP16_Optimizer(optimizer, static_loss_scale = args.loss_scale)
    #         linear_warmup_lr = get_linear_schedule_with_warmup(
    #             optimizer,
    #             num_warmup_steps = args.warmup_proportion * num_train_optimization_steps,
    #             num_training_steps = num_train_optimization_steps)

    else:
        # IMPORTANT: This is the original BERT Adam optimizer
        # optimizer = BertAdam(optimizer_grouped_parameters,
        #                      lr=args.learning_rate,
        #                      warmup=args.warmup_proportion,
        #                      t_total=num_train_optimization_steps)
        # IMPORTANT: To reproduce the old BertAdam specific behavior set correct_bias=False
        # Using the more recent AdamW, you need to add some things manually:
        # linear warmup scheduler for learning rate + gradient clipping
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate,
                          correct_bias = False)
        linear_warmup_lr = get_linear_schedule_with_warmup(optimizer,
                                                           num_warmup_steps = args.warmup_proportion * num_train_optimization_steps,
                                                           num_training_steps = num_train_optimization_steps)



    #############------------- PREPARE VALIDATION DATA ---------------#################

    # simply load the validation set and prepare it as BERT input (no corruption)
    logger.info('Creating examples from the validation split of the dataset.')
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                 tokenizer)
    logger.info('################# EVALUATION dataset #################')
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids_eval = torch.tensor([f.input_ids for f in eval_features], dtype = torch.long)
    all_input_mask_eval = torch.tensor([f.input_mask for f in eval_features], dtype = torch.long)
    all_segment_ids_eval = torch.tensor([f.segment_ids for f in eval_features], dtype = torch.long)
    all_label_ids_eval = torch.tensor([f.label_id for f in eval_features], dtype = torch.long)

    eval_data = TensorDataset(all_input_ids_eval, all_input_mask_eval, all_segment_ids_eval,
                              all_label_ids_eval)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler = eval_sampler,
                                 batch_size = args.eval_batch_size)

    ############------------- TRAINING LOOP ---------------#################
    logger.info('################# Starting TRAINING #################')
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # create Tensorboard writer
    writer_tb = SummaryWriter(log_dir = DIRECTORY_FOR_SAVING_OR_LOADING, flush_secs = 30,
                              filename_suffix = f'_training_{START_TIME}')

    for i in trange(int(args.num_train_epochs), desc = "Training epoch"):
        start_time_epoch = time.perf_counter()
        logger.info(f'Epoch {i + 1} has started.')
        ############------------- TRAINING ---------------#################
        # copy the parameters of the model
        logger.debug('Saving all named parameters before training the model...')
        frozen_parameters = {}
        for name, parameters in model.named_parameters():
            frozen_parameters[name] = copy.deepcopy(parameters.data)

        start_time_epoch_train = time.perf_counter()
        epoch_train_loss = train(train_loader = train_dataloader, model = model,
                                 optimizer = optimizer, lr_warmup = linear_warmup_lr,
                                 num_train_optimization_steps = num_train_optimization_steps)
        end_time_epoch_train = time.perf_counter()

        # check whether the parameters have been changed at all
        for name, p in model.named_parameters():
            # return updated = True, if any value for this layer is == True
            updated = (frozen_parameters[name] != p.data).any().cpu().detach().numpy()

            logger.debug(f"Layer '{name}' has been updated in epoch {i +1}? - {'yes' if updated else 'no'}")



        ############------------- VALIDATION ---------------#################
        # validate the trained model for loss + accuracy
        start_time_epoch_validate = time.perf_counter()
        epoch_validate_loss = validate(data_loader = eval_dataloader, model = model,
                                       tokenizer = tokenizer)
        end_time_epoch_validate = time.perf_counter()
        end_time_epoch = time.perf_counter()

        ############------------- LOGGING ---------------#################
        # add weights and gradients to tensorboard
        for parameter_name, values in model.named_parameters():
            if values.requires_grad is True:
                writer_tb.add_histogram(parameter_name, values, i)
                writer_tb.add_histogram(f'{parameter_name}.grad', values.grad, i)

        # add losses and evaluation metrics to tensorboard
        metric_dict = {'loss/training_loss': epoch_train_loss,
                       'loss/validation_loss': epoch_validate_loss,
                       'timings/total_epoch_runtime_min': round(
                           (end_time_epoch - start_time_epoch) / 60, 2),
                       'timings/train_runtime_min': round(
                           (end_time_epoch_train - start_time_epoch_train) / 60, 2),
                       'timings/validation_runtime_min': round(
                           (end_time_epoch_validate - start_time_epoch_validate) / 60, 2)}

        # add the metrics to tensorboard (all in a single file)
        for key, value in metric_dict.items():
            writer_tb.add_scalar(key, value, i)

        # make sure that the values are written to disk immediately
        writer_tb.flush()

        logger.info(f'Saved results of epoch {i + 1} to disk.')
        logger.info(
            f'Epoch {i + 1} of {int(args.num_train_epochs)} ran for {round((end_time_epoch - start_time_epoch_train) / 60, 2)} minutes.')

    ############------------- AFTER TRAINING IS COMPLETED ---------------#################
    logger.info('################# Finished TRAINING #################')

    # save the model (if not running distributed training)
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        # Save the trained model, configuration and tokenizer
        # save model.bin, config.json and vocab.txt to disk
        model.save_pretrained(os.path.join(DIRECTORY_FOR_SAVING_OR_LOADING, 'trained_model'),
                              save_config = True)
        tokenizer.save_vocabulary(os.path.join(DIRECTORY_FOR_SAVING_OR_LOADING, 'trained_model'))
        logger.info('Trained model saved to disk.')

    writer_tb.flush()
    writer_tb.close()

    END_TRAINING = time.perf_counter()
    logger.info(
        f'Training and validation (without LP metrics) took {round((END_TRAINING - START_TRAINING) / 60, 2)} minutes in total.')

    return model, tokenizer


def train(train_loader, model, optimizer, num_train_optimization_steps, lr_warmup):
    global global_step

    model.train()

    training_loss = torch.zeros(1, dtype = torch.float64)
    nb_tr_examples, nb_tr_steps = 0, 0

    loss_fct = CrossEntropyLoss()

    # IMPORTANT doublecheck that all layers are actually not frozen!
    for name, param in model.named_parameters():
        logger.debug(f'Layer {name} has requires_grad = {param.requires_grad}')

    for step, batch in enumerate(tqdm(train_loader, desc = "Iteration")):
        batch = tuple(t.to(device) for t in batch)  # unpack and send each tensor to device
        input_ids, input_mask, segment_ids, label_ids = batch  # unpack the batch

        # calculate predictions for current batch
        # do not automatically calculate the loss! (do not provide labels)
        logits = model(input_ids = input_ids, attention_mask = input_mask,
                       token_type_ids = segment_ids, labels = None,
                       output_hidden_states = False).logits

        # calculate loss, make sure that the tensors have correct dimensionality
        # view(-1) makes the tensor shape flexible
        loss = loss_fct(logits.view(-1, NUM_LABELS), label_ids.view(-1))

        model_output = model(input_ids = input_ids, attention_mask = input_mask,
                       token_type_ids = segment_ids, labels = input_ids,
                       output_hidden_states = False)
        #
        # model_output_with_loss = model(input_ids = input_ids, attention_mask = input_mask,
        #                token_type_ids = segment_ids, labels = None,
        #                output_hidden_states = False)

        # TODO calculate loss manually

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # calculate the backprogration with the loss
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()  # TODO decide whether to add gradient clipping here (part of pytorch_pretrained migration)  # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if loss > 1000:
            loss = loss.round()

        # add the loss value to the logging variables
        training_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        # this is always executed if gradient_accumulation_steps = 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = args.learning_rate * lr_warmup.get_lr(
                    global_step / num_train_optimization_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            # update the weights and set optimizer to zero
            optimizer.step()
            # logger.debug(f'Current learning rate is: {optimizer.param_groups[0]["lr"]}')
            lr_warmup.step()

            # IMPORTANT: debugging, look at gradients
            for parameter_name, values in model.named_parameters():
                if values.requires_grad is True:
                    logger.debug(f'{parameter_name}.grad: {values.grad}')

            optimizer.zero_grad()
            global_step += 1

        logger.debug(f'Accumulated training loss: {training_loss.item()}')

    # calculate mean training loss by dividing through number of batches
    training_loss_mean = training_loss / nb_tr_steps
    logger.info(f'Training loss (mean across all batches): {training_loss_mean}')

    return training_loss_mean


def validate(data_loader, model = None, tokenizer = None):

    if model is None and tokenizer is None:
        # in case evaluation is run independent of training
        # --> load a trained model and vocabulary that you have fine-tuned earlie
        model = BertForSequenceClassification.from_pretrained(
            os.path.join(DIRECTORY_FOR_SAVING_OR_LOADING, 'trained_model'), num_labels = NUM_LABELS
        )

        # folder needs to contain a config.json and a vocab.txt file
        tokenizer = BertTokenizer.from_pretrained(
            os.path.join(DIRECTORY_FOR_SAVING_OR_LOADING, 'trained_model'),
            do_lower_case = args.do_lower_case)

        model.to(device)
        model.eval()

        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                     tokenizer)
        logger.info('################# Starting EVALUATION #################')
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype = torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype = torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype = torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype = torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        data_loader = DataLoader(eval_data, sampler = eval_sampler,
                                 batch_size = args.eval_batch_size)

    # do typechecking to see that this was loaded correctly
    #assert type(model) == BertForSequenceClassification, 'Model was not loaded correctly.'
    #assert type(tokenizer) == BertTokenizer, 'Tokenizer was not loaded correctly.'

    eval_loss_accum = 0
    nb_eval_steps = 0

    ### intiate tensors used for metric calculation
    # use this to collect all logits across batches
    prediction_logits = torch.Tensor().float().to(device)
    # collect all true labels across batches
    all_true_labels = torch.Tensor().long().to(device)

    loss_fct = CrossEntropyLoss()

    model.eval()

    for input_ids, input_mask, segment_ids, label_ids in tqdm(data_loader, desc = "Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids = input_ids,
                           attention_mask = input_mask,
                           token_type_ids = segment_ids,
                           labels = None,
                           output_hidden_states = False).logits

            # create eval loss and other metric required by the task
            tmp_eval_loss = loss_fct(logits.view(-1, NUM_LABELS), label_ids.view(-1))

            # accumulate the loss over all batches (use mean() for multi-gpu case)
            eval_loss_accum += tmp_eval_loss.mean().item()
            logger.debug(f'accumulated eval_loss: {eval_loss_accum}')
            nb_eval_steps += 1
            # append the current predictions + true labels to the running tensor
            prediction_logits = torch.cat((prediction_logits, logits))
            all_true_labels = torch.cat((all_true_labels, label_ids.detach()))

    # calculate mean evaluation loss by dividing through number of batches
    eval_loss_mean = eval_loss_accum / nb_eval_steps
    logger.info(f'Evaluation loss (mean across all batches): {eval_loss_mean}')

    # make sure that dimensions are as expected
    # TODO decide whether probabilities are needed at all
    # get the position (0 or 1) for the largest value
    # calculate accuracy for the entire evaluation set
    # TODO add torchmetrics calculation here
    # evaluation_metrics = compute_metrics(prediction_prob, all_true_labels)

    return eval_loss_mean


def run_evaluation_after_training(model = None, tokenizer = None, data_loader = None):
    # load or access trained model and its tokenizer
    if model is None and tokenizer is None:
        logger.info(
            f'Loading trained model and tokenizer from: {os.path.join(DIRECTORY_FOR_SAVING_OR_LOADING, "trained_model")}')
        # in case evaluation on test set is run independent of training
        # --> load a trained model and vocabulary that you have fine-tuned earlie
        model = BertForSequenceClassification.from_pretrained(
            os.path.join(DIRECTORY_FOR_SAVING_OR_LOADING, 'trained_model'), num_labels = NUM_LABELS)

        # folder needs to contain a config.json and a vocab.txt file
        tokenizer = BertTokenizer.from_pretrained(
            os.path.join(DIRECTORY_FOR_SAVING_OR_LOADING, 'trained_model'),
            do_lower_case = args.do_lower_case)

    #############------------- CALCULATE LOSS ---------------#################

    model.to(device)
    model.eval()

    # load data if this is not run directly after training
    if data_loader is None:
        examples = None
        if args.do_eval:
            examples = processor.get_dev_examples(args.data_dir)

        if args.do_eval_on_test:
            examples = processor.get_test_examples(args.data_dir)
        assert examples is not None

        features = convert_examples_to_features(examples, label_list, args.max_seq_length,
                                                tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype = torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype = torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype = torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype = torch.long)

        tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                       all_label_ids)

        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler = sampler,
                                batch_size = args.eval_batch_size)

    if args.do_eval:
        # valid_loss = validate(model = model, tokenizer = tokenizer, data_loader = dataloader)
        link_prediction_metrics_dict = calculate_link_prediction_metrics(model, tokenizer,
                                                                         context = 'validation')  # link_prediction_metrics_dict['loss_on_validation_set'] = valid_loss

    if args.do_eval_on_test:
        test_loss = validate(model = model, tokenizer = tokenizer, data_loader = dataloader)
        link_prediction_metrics_dict = calculate_link_prediction_metrics(model, tokenizer,
                                                                         context = 'test')
        link_prediction_metrics_dict['loss_on_test_set'] = test_loss

    return link_prediction_metrics_dict


def calculate_link_prediction_metrics(model, tokenizer, context: str):
    """

    Parameters
    ----------
    model
    tokenizer
    context (str): needs to be "validation" or "test", this is used to save two differnt
        txt files collecting the calculated ranks, one for each context, i.e. dataset

    Returns
    -------
    result_dict (dict): Summarizes link prediction metrics results for all test triples.
    runtimes_test_triples_s (list): Contains the runtime in seconds for calculating the rank for
        each test triple.
    """
    ### Calculate link prediction metrics
    logger.info('*********** Start calculating link prediction metrics ***********')

    #############------------- PREPARE ALL VARIABLES ---------------#################
    assert context in ['validation',
                       'test'], "This function should be only used in contect of validation or test!"

    # load all triples of the dataset (train.tsv, dev.tsv, test.tsv)
    train_triples = processor.get_train_triples(args.data_dir)
    dev_triples = processor.get_dev_triples(args.data_dir)
    test_triples = processor.get_test_triples(args.data_dir)
    all_triples = train_triples + dev_triples + test_triples

    # create a set of strings
    # each string is the 3 parts of the triple joined by \t
    all_triples_str_set = set()
    for triple in all_triples:
        triple_str = '\t'.join(triple)
        all_triples_str_set.add(triple_str)

    # TODO maybe more efficient to initialize already in fin
    ranks_both = torch.Tensor().float().to(device)
    ranks_head = torch.Tensor().float().to(device)
    ranks_tail = torch.Tensor().float().to(device)

    hits_head = torch.empty(10).float().to(device)
    hits_tail = torch.empty(10).float().to(device)
    hits_both = torch.empty(10).float().to(device)

    top_ten_hit_count = 0

    test_triple_count = 0

    # IMPORTANT: select triples to go through depending on context
    if context == 'validation':
        logger.info('Running evaluation on the dataset split: dev.tsv')
        triples_to_evaluate = dev_triples
    if context == 'test':
        logger.info('Running evaluation on the dataset split: test.tsv')
        triples_to_evaluate = test_triples

    # Loop through all triples
    for test_triple in tqdm(triples_to_evaluate, desc = 'Evaluating triple'):
        logger.debug(
            f'Calculating rank for triple #{test_triple_count + 1} of {len(triples_to_evaluate)}')
        start_time_test_triple = time.perf_counter()
        head = test_triple[0]
        relation = test_triple[1]
        tail = test_triple[2]
        logger.debug(f'Current test triple: {head, relation, tail}')

        #############------------- CALCULATE RANK HEAD ---------------#################

        # create head_corrupt_list: the first item is the true triple
        # all remaining lines are triples that are incorrect, because the head entity is incorrect
        # filtered setting: exclude any triples that exist in the dataset!
        head_corrupt_list = [test_triple]
        for corrupt_ent in tqdm(entity_list):
            # do this for all entities except the actual head
            if corrupt_ent != head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                # only append the corrupted triple if it does not exist in the dataset
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(tmp_triple)
                if args.debug and len(head_corrupt_list) == 50:
                    break

        logger.info('######### Calculating rank head of current test triple #########')
        logger.debug(f'Length of head_corrupt list is: {len(head_corrupt_list)}')

        rank_head, test_triple_as_text, rank_head_metadata = calculate_rank_given_corrupt_list(
            corrupt_list = head_corrupt_list, index_of_triple = test_triple_count,
            type_of_rank = 'head', model = model, tokenizer = tokenizer, context = context)

        logger.info(f'Rank head for current triple: {rank_head.item() + 1}')

        # add this rank to the collecting variables
        plus_one = torch.ones(1).to(device)
        ranks_both = torch.cat((ranks_both, rank_head + plus_one))
        ranks_head = torch.cat((ranks_head, rank_head + plus_one))
        if rank_head < 10:
            top_ten_hit_count += 1

        #############------------- CALCULATE RANK TAIL ---------------#################

        # create tail_corrupt_list: the first item is the true triple
        # all remaining lines are triples that are incorrect, because the tail entity is incorrect
        # filtered setting: exclude any triples that exist in the dataset!
        tail_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = '\t'.join(tmp_triple)
                # only append the corrupted triple if it does not exist in the dataset
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(tmp_triple)
                if args.debug and len(tail_corrupt_list) == 10:
                    break

        logger.info('######### Calculating rank tail of current test triple #########')
        logger.debug(f'Length of tail_corrupt list is: {len(tail_corrupt_list)}')

        rank_tail, _, rank_tail_metadata = calculate_rank_given_corrupt_list(
            corrupt_list = tail_corrupt_list, index_of_triple = test_triple_count,
            type_of_rank = 'tail', model = model, tokenizer = tokenizer, context = context)

        ranks_both = torch.cat((ranks_both, rank_tail + plus_one))
        ranks_tail = torch.cat((ranks_tail, rank_tail + plus_one))
        logger.info(f'Rank tail for current triple: {rank_tail.item() + 1}')
        logger.info(f'Mean rank until now:  {torch.mean(ranks_both)}')

        if rank_tail < 10:
            top_ten_hit_count += 1
        logger.info(f"Hits@10 until now:  {top_ten_hit_count * 1.0 / len(ranks_both)}")

        #############------------- CALCULATE HITS@K ---------------#################

        # (original comment: this could be done more elegantly, but here you go)
        for hits_level in range(10):
            if rank_head <= hits_level:
                hits_both[hits_level] = 1.0
                hits_head[hits_level] = 1.0
            else:
                hits_both[hits_level] = 0.0
                hits_head[hits_level] = 0.0

            if rank_tail <= hits_level:
                hits_both[hits_level] = 1.0
                hits_tail[hits_level] = 1.0
            else:
                hits_both[hits_level] = 0.0
                hits_tail[hits_level] = 0.0

        end_time_test_triple = time.perf_counter()

        #############------------- LOG + SAVE ---------------#################

        # Save the current head + tail rank to disk
        file_name_for_saving = 'ranks_with_metadata_' + context + '_' + START_TIME + '.tsv'
        runtime_for_this_triple = round((end_time_test_triple - start_time_test_triple), 2)

        # create list object to add as row
        triples_and_ranks = [test_triple_count, test_triple, test_triple_as_text,
                             rank_head.item() + 1, rank_tail.item() + 1]

        row_to_add = triples_and_ranks + rank_head_metadata + rank_tail_metadata + [
            runtime_for_this_triple]
        row_to_add_with_tab = ''.join([str(x) + '\t' for x in row_to_add])

        # TODO alternatively, create a dataframe that can be returned and pickled
        # 'a' append to file, with 'w' each line is overwritten!
        with open(file_name_for_saving, 'a') as f:
            if test_triple_count == 0:
                column_names = ['test_triple_index', 'test_triple_IDs', 'test_triple_labels',
                                'rank_head', 'rank_tail', 'top_k_scores_head',
                                'top_k_entities_IDs_head', 'top_k_entities_labels_head',
                                'top_k_scores_tail', 'top_k_entities_IDs_tail',
                                'top_k_entities_labels_tail', 'runtime_sec']
                column_names_with_tab = ''.join([str(x) + '\t' for x in column_names])
                f.writelines(column_names_with_tab + '\n')
            f.writelines(row_to_add_with_tab + '\n')
        f.close()

        logger.info(f'Rank calculation for current triple took {runtime_for_this_triple} seconds.')
        test_triple_count += 1

    ### Calculate all link prediction metrics after having gone through all test triples
    # Log hits @1, @3, @5 and @10
    for i in [0, 2, 4, 9]:
        logger.info(f'Hits head @{i + 1}: {torch.mean(hits_head[i])}')
        logger.info(f'Hits tail @{i + 1}: {torch.mean(hits_tail[i])}')
        logger.info(f'Hits both @{i + 1}: {torch.mean(hits_both[i])}')
    logger.info(f'Mean rank head: {torch.mean(ranks_head)}')
    logger.info(f'Mean rank tail: {torch.mean(ranks_tail)}')
    logger.info(f'Mean rank both: {torch.mean(ranks_both)}')
    logger.info(f'Mean reciprocal rank head: {torch.mean(1. / ranks_head)}')
    logger.info(f'Mean reciprocal rank tail: {torch.mean(1. / ranks_tail)}')
    logger.info(f'Mean reciprocal rank both: {torch.mean(1. / ranks_both)}')
    # TODO add more metrics if required to compare with pykeen models!
    # adjusted mean rank by Berrendorf (2020)
    # adjusted mean rank index by Berrendorf (2020)
    # optimistic vs. pessimistic vs. realistic

    # each metric is a column
    result_dict = {'experiment_name': EXPERIMENT_NAME,
                   'mean_rank_both': torch.mean(ranks_both).item(),
                   'mean_rank_head': torch.mean(ranks_head).item(),
                   'mean_rank_tail': torch.mean(ranks_head).item(),
                   'mean_reciprocal_rank_both': torch.mean(1. / ranks_both).item(),
                   'mean_reciprocal_rank_head': torch.mean(1. / ranks_head).item(),
                   'mean_reciprocal_rank_tail': torch.mean(1. / ranks_tail).item(),
                   'hits_at_1': torch.mean(hits_both[0]).item(),
                   'hits_at_3': torch.mean(hits_both[2]).item(),
                   'hits_at_5': torch.mean(hits_both[4]).item(),
                   'hits_at_10': torch.mean(hits_both[9]).item()}

    return result_dict


def doublecheck_object_types(bla):
    assert type(bla) == list, 'This object is not a list'
    unique_classes = set(type(x).__name__ for x in bla)
    if isinstance(bla[0], InputFeatures):
        logger.info('Object is a list of InputFeatures.')  # assert bla[0].
    elif isinstance(bla[0], InputExample):
        logger.info('Object is a list of InputFeatures.')
        pass
    else:
        raise ValueError()


def calculate_rank_given_corrupt_list(corrupt_list: list, index_of_triple: int, type_of_rank: str,
                                      context: str, model, tokenizer):
    """
    Parameters
    ----------
    type_of_rank: can be head or tail, as this information is relevant for logging
    index_of_triple:
    corrupt_list: List where the first item is the true triple.
    model:
    tokenizer:

    Returns
    -------

    """
    assert type_of_rank in ['head', 'tail'], 'Type of rank is not recognized!'

    # TODO try to load from disk, except: create them yourself on the fly
    ### convert string triples to BERT input feature vectors
    # this accesses the labels as text for each relation and entity
    tmp_examples = processor._create_examples(corrupt_list, set_type = "test",
                                              data_dir = args.data_dir)
    test_triple_as_text = f'{tmp_examples[0].text_a} | {tmp_examples[0].text_b} | {tmp_examples[0].text_c}'
    logger.debug(f'Current test triple as text is: {test_triple_as_text}')
    # logger.debug(f'Size of tmp_examples: {sys.getsizeof(tmp_examples)}')

    # TODO check whether file exists on disk
    # if not, throw a warning and create it from scratch (this should be avoided)

    tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length,
                                                tokenizer, print_info = False)

    # logger.debug(f'Size of tmp_features: {sys.getsizeof(tmp_features)}'
    all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype = torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype = torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype = torch.long)
    all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype = torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # logger.debug(f'Size of TensorDataset: {sys.getsizeof(eval_data)}')

    # Run prediction for temp data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler = eval_sampler,
                                 batch_size = args.eval_batch_size)
    # logger.debug(f'Size of DataLoader: {sys.getsizeof(eval_dataloader)}')
    model.eval()

    prediction_logits = torch.Tensor().float().to(device)

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        # label_ids not relevant here as they are always true == 1
        # label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids = input_ids, attention_mask = input_mask,
                           token_type_ids = segment_ids, labels = None,
                           output_hidden_states = False).logits

        logger.debug(f'Logits: {logits}')

        prediction_logits = torch.cat((prediction_logits, logits))

    # get the dimension corresponding to the label that indicates a true triple
    # label 1 = true triple
    # TODO doublecheck that this is a number
    position_of_correct_label = all_label_ids[0]
    plausibility_scores_logits = prediction_logits[:, position_of_correct_label]

    # TODO reduce overhead: make this logging optional or do it every 100 test triples
    ### IMPORTANT DEBUGGING: look at whether the prediction logits make sense!
    # This shows whether the model has actually learned the link prediction task!
    # report difference between mean probability value and
    plausibility_scores_probs = F.softmax(plausibility_scores_logits, dim = 0)
    uniform_probability = 1 / len(corrupt_list)
    logger.debug(
        f'Absolute difference between mean probability score and uniform probability is: {abs(torch.mean(plausibility_scores_probs).item() - uniform_probability)}')

    # write the predictions as probability values to tensorboard
    # (probabilities are more intuitive to understand than logits)
    writer_tb.add_histogram(f'plausibility_scores_{type_of_rank}', plausibility_scores_probs,
                            index_of_triple)

    # retrieve sorted descending order of plausibility predictions
    sorted_values, sorted_indices = torch.sort(plausibility_scores_logits, descending = True)

    # extract the k highest probabilities and the corresponding entity IDs and labels
    top_k = 10
    top_k_scores = list(sorted_values[0:top_k].cpu().numpy())
    top_k_entities_indices = list(sorted_indices[0:top_k].cpu().numpy())

    # retrieve the dataset-specific IDs + natural language labels for the top_k entities
    if type_of_rank == 'head':
        # use first item = head
        top_k_entities_dataset_IDs = [corrupt_list[index][0] for index in top_k_entities_indices]
        top_k_entities_labels = [tmp_examples[index].text_a for index in top_k_entities_indices]
    elif type_of_rank == 'tail':
        # use third item = tail
        top_k_entities_dataset_IDs = [corrupt_list[index][2] for index in top_k_entities_indices]
        top_k_entities_labels = [tmp_examples[index].text_c for index in top_k_entities_indices]

    # Retrieve the rank of the correct triple (first position in the list)
    # remember: within the data, the first item was the only correct triple, the rest
    # were corrupted ones, i.e. created by replacing the head entity
    # is equivalent to: np.where(sorted_indices.numpy() == 0)[0]
    # TODO just use np.where ?
    rank = torch.where(sorted_indices == 0)[0]  # result is a tuple, take first item

    return rank, test_triple_as_text, [top_k_scores, top_k_entities_dataset_IDs,
                                       top_k_entities_labels]


def get_KG_BERT_embeddings(context: str):
    # load the trained model
    # when running predictions with the model, set output_hidden_states = True
    raise NotImplementedError()


def preprocess_and_save_triples(context: str, dataset_name: str, tokenizer = None):
    if tokenizer is None:
        # folder needs to contain a config.json and a vocab.txt file
        tokenizer = BertTokenizer.from_pretrained(args.bert_model,
            do_lower_case = args.do_lower_case,
            vocab_file = os.path.join(BASE_PATH_HOST, 'data/interim/KG_and_LM/vocab.txt'))

    assert context in ['validation',
                       'test'], "This function should be only used in context of validation or test!"

    # load all triples of the dataset (train.tsv, dev.tsv, test.tsv)
    train_triples = processor.get_train_triples(args.data_dir)
    dev_triples = processor.get_dev_triples(args.data_dir)
    test_triples = processor.get_test_triples(args.data_dir)
    all_triples = train_triples + dev_triples + test_triples

    # create a set of strings
    # each string is the 3 parts of the triple joined by \t
    all_triples_str_set = set()
    for triple in all_triples:
        triple_str = '\t'.join(triple)
        all_triples_str_set.add(triple_str)

    # IMPORTANT: select triples to go through depending on context
    if context == 'validation':
        logger.info('Running evaluation on the dataset split: dev.tsv')
        triples_to_evaluate = dev_triples
    if context == 'test':
        logger.info('Running evaluation on the dataset split: test.tsv')
        triples_to_evaluate = test_triples

    test_triple_count = 0

    for test_triple in tqdm(triples_to_evaluate, desc = 'Evaluating triple'):
        start_preprocess_triple = time.perf_counter()
        logger.debug(
            f'Calculating rank for triple #{test_triple_count + 1} of {len(triples_to_evaluate)}')
        head = test_triple[0]
        relation = test_triple[1]
        tail = test_triple[2]
        logger.debug(f'Current test triple: {head, relation, tail}')

        #############------------- SAVE HEAD InputFeatures ---------------#################

        # create head_corrupt_list: the first item is the true triple
        # all remaining lines are triples that are incorrect, because the head entity is incorrect
        # filtered setting: exclude any triples that exist in the dataset!
        head_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            # do this for all entities except the actual head
            if corrupt_ent != head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                # only append the corrupted triple if it does not exist in the dataset
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(
                        tmp_triple)  # if args.debug and len(head_corrupt_list) == 10:  #     break

        ### convert string triples to BERT input feature vectors
        # this accesses the labels as text for each relation and entity
        tmp_examples = processor._create_examples(head_corrupt_list, set_type = "test",
                                                  data_dir = args.data_dir)
        test_triple_as_text = f'{tmp_examples[0].text_a} | {tmp_examples[0].text_b} | {tmp_examples[0].text_c}'
        logger.debug(f'Current test triple as text is: {test_triple_as_text}')

        tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length,
                                                    tokenizer, print_info = False)

        # IMPORTANT: save the preprocessed corrupt list (as InputFeatures) to disk
        # in first iteration, create an empty dict, then add an item each time this function is called
        if test_triple_count == 0:
            dict_for_saving_InputFeatures = {}  # determine length of dataset for deciding on checks  # max_index depends on dataset, using context

        # key = something to identify the test triple
        #    option 1: ''.join(corrupt_list[0]) --> safe option: concatenated string of triple IDs
        #    option 2: index_of_triple + 1 --> integer corresponding to row number in unshuffled dataset
        # value = tuple of size two for head and tail corruption
        # value_alt = a list so I can append to it
        dict_key = ''.join(head_corrupt_list[0])
        logger.debug(f'Current dict_key: {dict_key}')

        # IMPORTANT save in first position of list
        dict_for_saving_InputFeatures[dict_key] = [tmp_features]

        #############------------- SAVE TAIL InputFeatures ---------------#################

        # create tail_corrupt_list: the first item is the true triple
        # all remaining lines are triples that are incorrect, because the tail entity is incorrect
        # filtered setting: exclude any triples that exist in the dataset!
        tail_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = '\t'.join(tmp_triple)
                # only append the corrupted triple if it does not exist in the dataset
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(
                        tmp_triple)  # if args.debug and len(tail_corrupt_list) == 10:  #     break

        ### convert string triples to BERT input feature vectors
        # this accesses the labels as text for each relation and entity
        tmp_examples = processor._create_examples(head_corrupt_list, set_type = "test",
                                                  data_dir = args.data_dir)
        tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length,
                                                    tokenizer, print_info = False)

        # IMPORTANT save in second position of list
        dict_for_saving_InputFeatures[dict_key] = dict_for_saving_InputFeatures.get(dict_key) + [
            tmp_features]

        logger.debug(f'Current # of keys of dict: {len(dict_for_saving_InputFeatures.keys())}')

        end_preprocess_triple = time.perf_counter()
        logger.info(
            f'Processing of triple {test_triple_count + 1} took {round((end_preprocess_triple - start_preprocess_triple) / 60, 2)} minutes.')
        test_triple_count += 1

    logger.info('Went through entire dataset!')

    torch.save(dict_for_saving_InputFeatures,
               f'{dataset_name}_{context}_triples_preprocessed_as_InputFeatures.pt')

    # IMPORTANT after all iterations, check that the object is as expected

    for key, value in tqdm(dict_for_saving_InputFeatures.items(),
                           desc = 'Running assertion checks on all dict entries...'):
        logger.debug(f'Current key: {key}')
        # each value should be a list of length 2
        assert type(value) == list
        assert len(value) == 2, 'More than two entries in this row!'
        # check that the two entries are different
        assert value[0] != value[1], 'Both entries are the same!'

    # assert that all keys are contained in the set of dev/test triples
    if context == 'validation':
        dev_triples_set = set()
        for triple in dev_triples:
            triple_str = ''.join(triple)
            dev_triples_set.add(triple_str)
        assert set(
            dict_for_saving_InputFeatures.keys()) == dev_triples_set, 'Not all validation triples are in the resulting dict!'

    if context == 'test':
        test_triples_set = set()
        for triple in test_triples:
            triple_str = ''.join(triple)
            test_triples_set.add(triple_str)
        assert set(
            dict_for_saving_InputFeatures.keys()) == test_triples_set, 'Not all test triples are in the resulting dict!'

    # IMPORTANT now try to save this reproducibably to disk
    torch.save(dict_for_saving_InputFeatures,
               f'FB15k-237_{context}_triples_preprocessed_as_InputFeatures.pt')

    # torch.load('preprocessed_test_triples_tmp_features.pt')


START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")

parser = argparse.ArgumentParser(
    description = 'Run experiments with Wikidata5M as the data source and '
                  'a fine-tuned BERT, i.e. KG-BERT as the model.')

## Required parameters (set default to None)
# TODO change back to None, provided manual defaults for debugging

# debugging example: FB15k237
parser.add_argument("--data_dir",
                    default = '/home/lena/git/master_thesis_bias_in_NLP/code_from_other_papers/Yao_KG_BERT/data/FB15k-237',
                    type = str, required = True,
                    help = "The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default = 'bert-base-cased', type = str, required = True,
                    help = "Bert pre-trained model selected in the list: bert-base-uncased, "
                           "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                           "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument('-n', '--name', type = str,
                    help = 'Experiment name, - if not doing preprocessing- a folder with this name'
                           ' will be created  or loaded from in the results/KG_and_LM folder.')

## Other parameters
parser.add_argument('-d', "--debug", action = 'store_true',
                    help = "Add this flag when debugging. Will adapt parameters such that the model runs way faster.")
parser.add_argument("--cache_dir", default = "", type = str,
                    help = "Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default = 128, type = int,
                    help = "The maximum total input sequence length after WordPiece tokenization. \n"
                           "Sequences longer than this will be truncated, and sequences shorter \n"
                           "than this will be padded.")
parser.add_argument("--do_train", action = 'store_true',
                    help = "Whether to run training and validation.")
parser.add_argument("--do_eval", action = 'store_true',
                    help = "Whether to run link prediction metric calculation on the vaidation set. "
                           "Uses the trained model in the folder")
parser.add_argument("--do_eval_on_test", action = 'store_true',
                    help = "Whether to run eval on the test set.")
parser.add_argument("--do_preprocessing_val", action = 'store_true',
                    help = "Whether to preprocess the validation triples and store them")
parser.add_argument("--do_preprocessing_test", action = 'store_true')
parser.add_argument("--do_lower_case", action = 'store_true',
                    help = "Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size", default = 32, type = int,
                    help = "Total batch size for training.")
parser.add_argument("--eval_batch_size", default = 8, type = int,
                    help = "Total batch size for eval.")
parser.add_argument("--learning_rate", default = 5e-5, type = float,
                    help = "The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default = 3.0, type = float,
                    help = "Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default = 0.1, type = float,
                    help = "Proportion of training to perform linear learning rate warmup for. "
                           "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda", action = 'store_true',
                    help = "Whether not to use CUDA when available")
parser.add_argument("--local_rank", type = int, default = -1,
                    help = "Specify local_rank for distributed training on gpus. If -1, distributed training is disabled.")
parser.add_argument('--seed', type = int, default = 42, help = "random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type = int, default = 1,
                    help = "Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16', action = 'store_true',
                    help = "Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale', type = float, default = 0,
                    help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                           "0 (default value): dynamic loss scaling.\n"
                           "Positive power of 2: static loss scaling value.\n")
args = parser.parse_args()

if args.do_preprocessing_val or args.do_preprocessing_test:
    start_preprocessing = time.perf_counter()
    # set the context variable
    if args.do_preprocessing_val:
        context = 'validation'
    elif args.do_preprocessing_test:
        context = 'test'

    # set the working directory for saving logger + result files
    if args.debug:
        EXPERIMENT_NAME = 'DEBUGGING_' + START_TIME + f'preprocessing_{context}'
        # test + dev have 100 examples, training has 500
        args.data_dir = os.path.join(args.data_dir, 'for_debugging')
    else:
        EXPERIMENT_NAME = START_TIME + f'_preprocessing_{context}'

    DIRECTORY_FOR_SAVING_OR_LOADING = os.path.join(BASE_PATH_HOST, 'data/interim/KG_and_LM',
                                                   EXPERIMENT_NAME)
    if not os.path.exists(DIRECTORY_FOR_SAVING_OR_LOADING):
        os.makedirs(DIRECTORY_FOR_SAVING_OR_LOADING)
    os.chdir(DIRECTORY_FOR_SAVING_OR_LOADING)

    # save the currently running script file for later reference
    file_name_script = 'script_' + EXPERIMENT_NAME + '.py'
    source_path = os.path.join(BASE_PATH_HOST, 'src/CLI_scripts', __file__)
    shutil.copy(src = source_path, dst = os.path.join(os.getcwd(), file_name_script))

    # initialize logger
    logger_file_name = f'log_preprocessing_{context}_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'
    logger = initialize_my_logger(file_name = logger_file_name, level = logging.DEBUG)
    logger.info(f'Saving everything in folder: {DIRECTORY_FOR_SAVING_OR_LOADING}')

    # get global variables needed for preprocessing
    processor = KGProcessor()
    label_list = processor.get_labels(args.data_dir)
    entity_list = processor.get_entities(args.data_dir)

    preprocess_and_save_triples(context = context, dataset_name = 'FB15k-237')

    end_preprocessing = time.perf_counter()
    logger.info(
        f'Processing of {context} triple set took {round((end_preprocessing - start_preprocessing) / 60, 2)} minutes.')
    # end the script here, do not run any training or evaluation
    sys.exit()

# only save certain things when actually training a model
if args.do_train:
    # set dataset paths and experiment name
    if args.debug:
        EXPERIMENT_NAME = 'DEBUGGING_' + START_TIME + '_' + args.name
        # test + dev have 100 examples, training has 500
        args.data_dir = os.path.join(args.data_dir, 'for_debugging')
    else:
        EXPERIMENT_NAME = START_TIME + '_' + args.name

    # create path for saving all the result files
    DIRECTORY_FOR_SAVING_OR_LOADING = os.path.join(BASE_PATH_HOST, 'results/KG_and_LM',
                                                   EXPERIMENT_NAME)

    # create directory and then use it as working directory
    if os.path.exists(DIRECTORY_FOR_SAVING_OR_LOADING) and os.listdir(
            DIRECTORY_FOR_SAVING_OR_LOADING) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(
            DIRECTORY_FOR_SAVING_OR_LOADING))
    if not os.path.exists(DIRECTORY_FOR_SAVING_OR_LOADING):
        os.makedirs(DIRECTORY_FOR_SAVING_OR_LOADING)

    # change working directory to respective folder
    os.chdir(DIRECTORY_FOR_SAVING_OR_LOADING)

    # save the currently running script file for later reference
    file_name_script = 'script_' + EXPERIMENT_NAME + '.py'
    source_path = os.path.join(BASE_PATH_HOST, 'src/CLI_scripts', __file__)
    shutil.copy(src = source_path, dst = os.path.join(os.getcwd(), file_name_script))

    # save the argparse arguments to disk
    save_argparse_obj_to_disk(argparse_namespace = args)

    if args.do_eval is False:
        logger_file_name = f'log_train_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'

    if args.do_eval:
        logger_file_name = f'log_train_eval_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'

    if args.do_eval is False and args.do_eval_on_test:
        logger_file_name = f'log_train_test_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'

    # configure the logging to stdout and file
    logger = initialize_my_logger(file_name = logger_file_name, level = logging.DEBUG)

    logger.info(f'Saving everything in folder: {DIRECTORY_FOR_SAVING_OR_LOADING}')

    if args.debug:
        logger.info('DEBUGGING MODE: Using a very small subset of FB15k237.')

### IMPORTANT: NOT TRAINING, ONLY EVALUATION TO CALCULATE LINK PREDICTION METRICS
elif args.do_train is False:

    # if not training, save everything to the folder where the model is loaded from
    EXPERIMENT_NAME = args.name
    DIRECTORY_FOR_SAVING_OR_LOADING = os.path.join(BASE_PATH_HOST, 'results/KG_and_LM',
                                                   EXPERIMENT_NAME)
    if not os.path.exists(DIRECTORY_FOR_SAVING_OR_LOADING):
        raise FileNotFoundError(
            'Directory does not exist! Experiment name must refer to an existing'
            'directory inside results/KG_and_LM!')
    # change working directory to respective folder
    os.chdir(DIRECTORY_FOR_SAVING_OR_LOADING)

    # configure the logging file name
    if args.do_eval and args.do_eval_on_test is False:
        logger_file_name = f'log_valid_{socket.gethostname()}_' + START_TIME + '.txt'
    elif args.do_eval_on_test and args.do_eval is False:
        logger_file_name = f'log_test_{socket.gethostname()}_' + START_TIME + '.txt'
    else:
        raise NotImplementedError('This combination of run train/eval/test is not accounted for!')

    logger = initialize_my_logger(file_name = logger_file_name, level = logging.DEBUG,
                                  file_mode = 'w')

    logger.info(f'Loading model from folder: {DIRECTORY_FOR_SAVING_OR_LOADING}')

    if args.debug:
        # test + dev have 100 examples, training has 500
        args.data_dir = os.path.join(args.data_dir, 'for_debugging')
        logger.info('DEBUGGING MODE: Using a very small subset of FB15k237.')

# set the correct CUDA device, check for number of devices
if args.local_rank == -1 or args.no_cuda:
    # important: if no_cuda is enabled, use CPU even though GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # IMPORTANT if n_gpu > 1 this enables  simple distributed training
    # using model = torch.nn.DataParallel(model)
    n_gpu = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    logger.info('Distributed training using torch.distributed backend.')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend = 'nccl')

logger.info('################# DEVICE INFORMATION #################')
logger.info(f'Device used: {device}')
logger.info(f"CUDA is used: {torch.cuda.is_available()}")
logger.info(f'Using {n_gpu} device(s).')
if n_gpu > 1:
    logger.info('Using torchs DataParallel mode for the model.')
    logger
logger.info(f'Using torch.distributed: {bool(args.local_rank != -1)}')
logger.info(f'Using half-precision float16 datatype: {args.fp16}')

# set the random seeds
args.seed = random.randint(1, 200)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# TODO decide whether to use this: torch.backends.cudnn.deterministic = True

if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

### load KG Processor, entity list and label list of the dataset
processor = KGProcessor()

# access to binary labels + entities.txt file to create a list object
label_list = processor.get_labels(args.data_dir)
NUM_LABELS = len(label_list)
entity_list = processor.get_entities(args.data_dir)

global_step = 0  # counts steps across all functions: train, validate, test

# IMPORTANT: Training starts here
if args.do_train:
    trained_model, loaded_tokenizer = train_and_validate()

if args.do_eval:
    writer_tb = SummaryWriter(log_dir = DIRECTORY_FOR_SAVING_OR_LOADING, flush_secs = 30,
                              filename_suffix = f'_validation_{START_TIME}')
    START_VALIDATION = time.perf_counter()

    if args.do_train:
        # TODO figure out how to pass the eval_dataloader form train_and_validate() here
        valid_results_dict = run_evaluation_after_training(trained_model, loaded_tokenizer)
    else:
        # if running validation without training, load model + tokenizer from working directory
        valid_results_dict = run_evaluation_after_training()

    # dict keys = dataframe columns
    valid_only_results_df = pd.DataFrame([valid_results_dict])

    # Save all metrics to a file
    file_name_valid_only_results = 'link_prediction_results_valid.csv'
    valid_only_results_df.to_csv(file_name_valid_only_results)
    logger.info('Saved validation results as CSV file to working directory.')

    ### IMPORTANT save metrics dict to hparams in tensorboard for model selection
    # use link prediction metrics for the hparams view in tensorboard
    # (they are more insightful than loss and timings)
    hparams_dict = args.__dict__
    # TODO careful, these items always need to match the current argparse options!!!
    items_to_remove = ['data_dir', 'name', 'debug', 'cache_dir', 'do_train', 'do_eval',
                       'do_eval_on_test', 'do_preprocessing_val', 'do_preprocessing_test',
                       'no_cuda', 'local_rank']
    for item in items_to_remove:
        hparams_dict.pop(item)

    # make metric_dict purely numeric, this is expected by tensorboard!
    valid_results_dict.pop('experiment_name')
    for value in valid_results_dict.values():
        assert type(
            value) == float, 'Doublecheck, metric values need to be numeric for use with add_hparams!'

    # TODO add ranges/options for all discrete hparams keys
    # option: hparam_domain_discrete =
    writer_tb.add_hparams(hparam_dict = hparams_dict, metric_dict = valid_results_dict,
                          run_name = 'tb_hparams_validation' + EXPERIMENT_NAME)
    writer_tb.flush()
    writer_tb.close()

    END_VALIDATION = time.perf_counter()
    logger.info(
        f'Calculation of link prediction metrics on validation set took {round((END_VALIDATION - START_VALIDATION) / 60, 2)} minutes in total.')

# IMPORTANT: Running evaluation on test set starting from here
if args.do_eval_on_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    writer_tb = SummaryWriter(log_dir = DIRECTORY_FOR_SAVING_OR_LOADING, flush_secs = 30,
                              filename_suffix = f'_test_{START_TIME}')
    START_TEST = time.perf_counter()

    #preprocess_and_save_triples(context = 'test')
    if args.do_train:
        test_results_dict = run_evaluation_after_training(trained_model, loaded_tokenizer)
    else:
        # if running test without training, load model + tokenizer from working directory
        test_results_dict = run_evaluation_after_training()

    # dict keys = dataframe columns
    test_results_df = pd.DataFrame([test_results_dict])

    # Save all metrics to a file
    file_name_test_results = 'link_prediction_results_test.csv'
    test_results_df.to_csv(file_name_test_results)
    logger.info('Saved test results as CSV file to working directory.')

    writer_tb.flush()
    writer_tb.close()

    END_TEST = time.perf_counter()
    logger.info(
        f'Calculation of link prediction metrics on test set took {round((END_TEST - START_TEST) / 60, 2)} minutes in total.')

logger.info(f'Finished running the script at: {datetime.now().strftime("%d.%m.%Y %H:%M")}')
