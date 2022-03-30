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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

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

# create a logger from the module
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
                    for j in range(5):
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
                    for j in range(5):
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
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # tokenize the head entity, outputs a list of strings
        tokens_a = tokenizer.tokenize(example.text_a)

        # set tokens to None because they will be changed?
        tokens_b = None
        tokens_c = None

        # this is only executed if both objects are not None
        if example.text_b and example.text_c:
            # tokenize relation and tail entity with the tokenizer
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # TODO Account for [CLS], [SEP], [SEP], [SEP] with "- 4" with max_seq_length - 4
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

        # print information about the first 5 examples of the dataset
        if ex_index < 1 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")
    parser = argparse.ArgumentParser()

    ## Required parameters (set default to None)
    parser.add_argument("--data_dir", default = None, type = str, required = True,
                        help = "The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default = None, type = str, required = True,
                        help = "Bert pre-trained model selected in the list: bert-base-uncased, "
                               "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                               "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default = None, type = str, required = True,
                        help = "The name of the task to train.")
    parser.add_argument("--output_dir", type = str, required = True,
                        help = "The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument('-d', "--debug", action = 'store_true',
                        help = "Add this flag when debugging. Will adapt parameters such that the model runs way faster.")
    parser.add_argument("--cache_dir", default = "", type = str,
                        help = "Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default = 128, type = int,
                        help = "The maximum total input sequence length after WordPiece tokenization. \n"
                               "Sequences longer than this will be truncated, and sequences shorter \n"
                               "than this will be padded.")
    parser.add_argument("--do_train", action = 'store_true', help = "Whether to run training.")
    parser.add_argument("--do_eval", action = 'store_true',
                        help = "Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action = 'store_true',
                        help = "Whether to run eval on the test set.")
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
                        help = "local_rank for distributed training on gpus")
    parser.add_argument('--seed', type = int, default = 42, help = "random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1,
                        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action = 'store_true',
                        help = "Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                               "0 (default value): dynamic loss scaling.\n"
                               "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type = str, default = '',
                        help = "Can be used for distant debugging.")
    parser.add_argument('--server_port', type = str, default = '',
                        help = "Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address = (args.server_ip, args.server_port), redirect_output = True)
        ptvsd.wait_for_attach()

    if args.do_eval is True:
        raise ValueError("doublecheck logging setting for this!")

    # set working directory
    if args.do_train:
        if args.debug:
                EXPERIMENT_NAME = 'DEBUGGING_' + START_TIME + '_minimal_change_script'
                # test + dev have 100 examples, training has 500
                args.data_dir = os.path.join(args.data_dir, 'for_debugging')
        else:
            EXPERIMENT_NAME = START_TIME + '_minimal_change_script'

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

        if args.do_train and args.do_eval is False:
            logger_file_name = f'log_train_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'

        if args.do_train and args.do_eval:
            logger_file_name = f'log_train_eval_{socket.gethostname()}_' + EXPERIMENT_NAME + '.txt'

        # configure the logging
        format_long = '%(asctime)s - %(levelname)s - %(filename)s/%(funcName)s: %(message)s'
        format_short = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(format = format_long, level = logging.DEBUG, datefmt = "%d.%m.%Y %H:%M:%S",
                            handlers = [logging.FileHandler(logger_file_name, mode = 'a'),
                                logging.StreamHandler(sys.stdout)])
        logger = logging.getLogger()

        logger.info(f'Saving everything in folder: {DIRECTORY_FOR_SAVING_OR_LOADING}')

    if args.do_predict:
        os.chdir(args.output_dir)

        logger_file_name = f'log_evaluation_test_set_{START_TIME}_{socket.gethostname()}.txt'

        # configure the logging
        format_long = '%(asctime)s - %(levelname)s - %(filename)s/%(funcName)s: %(message)s'
        format_short = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(format = format_long, level = logging.DEBUG, datefmt = "%d.%m.%Y %H:%M:%S",
                            handlers = [logging.FileHandler(logger_file_name, mode = 'a'),
                                logging.StreamHandler(sys.stdout)])
        logger = logging.getLogger()

        logger.info(f'Saving everything in folder: {args.output_dir}')

    if args.debug:
        logger.info('DEBUGGING MODE: Using a very small subset of FB15k237.')

    # set the correct CUDA device, check for number of devices
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
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
    logger.info(f'Using distributed training: {bool(args.local_rank != -1)}')
    logger.info(f'Using half-precision float16 datatype: {args.fp16}')

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # divide batch size by gradient accumulation steps
    # this is not relevant for the results, usually = 1!
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # set the random seeds
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # create a KGProcessor + task name
    processors = {"kg": KGProcessor, }

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # create a KGProcessor object
    processor = processors[task_name]()

    # access to entities.txt file to create a list object
    # IMPORTANT access to file entities.txt
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)
    entity_list = processor.get_entities(args.data_dir)

    #############------------- LOAD MODEL ---------------#################
    #
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = args.do_lower_case)
    #
    # # get training examples + create NEGATIVE/CORRUPT triples for each of it
    #
    # train_examples = None
    # num_train_optimization_steps = 0
    # if args.do_train:
    #     #############------------- START TRAINING ---------------#################
    #
    #     START_TRAINING = time.perf_counter()
    #     logger.info(
    #         'Creating examples with negative sampling from the training split of the dataset.')
    #     train_examples = processor.get_train_examples(args.data_dir)
    #     num_train_optimization_steps = int(
    #         len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    #     if args.local_rank != -1:
    #         num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    #
    #
    #  # parameters will be loaded as float 32
    #     # Prepare model, download BERT for sequence classification
    #     # create cache folder so the model is only downloaded when the script is run for the first time on the machine
    #     cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(TRANSFORMERS_CACHE),
    #                                                                    'distributed_{}'.format(
    #                                                                        args.local_rank))
    #
    #     model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir = cache_dir,
    #                                                       num_labels = num_labels)
    #
    #     # if specified, cast the model to float 16 datatype
    #     if args.fp16:
    #         model.half()
    #
    #     model.to(device)
    #
    # # if specified, enable distributed training on GPUs using torch.nn.DataParallel(model)
    # if args.local_rank != -1:
    #     try:
    #         from apex.parallel import DistributedDataParallel as DDP
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #
    #     model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)  # model = torch.nn.parallel.data_parallel(model)
    #
    # # Prepare optimizer
    # param_optimizer = list(model.named_parameters())
    #
    # # add all parameters except decay?
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # # TODO unclear: set weight decay to zero?
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}]
    #
    # # use other optimizers if using float16 data type
    # if args.fp16:
    #     raise NotImplementedError('Using fp16, doublecheck all settings!')
    #     # try:
    #     #     from apex.optimizers import FP16_Optimizer
    #     #     from apex.optimizers import FusedAdam
    #     # except ImportError:
    #     #     raise ImportError(
    #     #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #     #
    #     # optimizer = FusedAdam(optimizer_grouped_parameters, lr = args.learning_rate,
    #     #                       bias_correction = False, max_grad_norm = 1.0)
    #     # if args.loss_scale == 0:
    #     #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale = True)
    #     # else:
    #     #     optimizer = FP16_Optimizer(optimizer, static_loss_scale = args.loss_scale)
    #     # warmup_linear = WarmupLinearSchedule(warmup = args.warmup_proportion,
    #     #                                      t_total = num_train_optimization_steps)
    #
    # else:
    #     # IMPORTANT: This is the original BERT Adam optimizer
    #     # optimizer = BertAdam(optimizer_grouped_parameters,
    #     #                      lr=args.learning_rate,
    #     #                      warmup=args.warmup_proportion,
    #     #                      t_total=num_train_optimization_steps)
    #     # IMPORTANT: To reproduce the old BertAdam specific behavior set correct_bias=False
    #     # Using the more recent AdamW, you need to add some things manually:
    #     # linear warmup scheduler for learning rate + gradient clipping
    #     optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate,
    #                       correct_bias = False)
    #     linear_warmup_lr = get_linear_schedule_with_warmup(optimizer,
    #                                                        num_warmup_steps = args.warmup_proportion * num_train_optimization_steps,
    #                                                        num_training_steps = num_train_optimization_steps)
    #
    #
    global_step = 0  # ?
    # nb_tr_steps = 0
    # tr_loss = 0  # accumulate training loss?

    # DO THE TRAINING!!!
    if args.do_train:

        #############------------- PREPARE TRAINING DATA ---------------#################

        train_features = convert_examples_to_features(train_examples, label_list,
            args.max_seq_length, tokenizer)
        logger.info('################# TRAINING dataset #################')
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        # create 4 separate variables out of train_features
        # store everything in a long Tensor
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype = torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype = torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype = torch.long)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype = torch.long)

        # wrap all tensors into a torch.TensorDataset
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)  # default: replacement = False
        else:
            train_sampler = DistributedSampler(train_data)

        # Pytorch data_loader, not shuffled (RandomSampler does that)
        train_dataloader = DataLoader(train_data, sampler = train_sampler,
                                      batch_size = args.train_batch_size)

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
        all_input_mask_eval = torch.tensor([f.input_mask for f in eval_features],
                                           dtype = torch.long)
        all_segment_ids_eval = torch.tensor([f.segment_ids for f in eval_features],
                                            dtype = torch.long)
        all_label_ids_eval = torch.tensor([f.label_id for f in eval_features], dtype = torch.long)

        eval_data = TensorDataset(all_input_ids_eval, all_input_mask_eval, all_segment_ids_eval,
                                  all_label_ids_eval)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler = eval_sampler,
                                     batch_size = args.eval_batch_size)

        #############------------- TRAINING LOOP ---------------#################

        # create Tensorboard writer
        writer_tb = SummaryWriter(log_dir = DIRECTORY_FOR_SAVING_OR_LOADING, flush_secs = 30,
                                  filename_suffix = f'_training_{START_TIME}')

        for i in trange(int(args.num_train_epochs), desc = "Epoch"):
            start_time_epoch = time.perf_counter()
            logger.info(f'Epoch {i + 1} has started.')
            # note that global_step is not set to zero
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            model.train()  # set model to train mode
            ############------------- TRAINING ---------------#################
            start_time_epoch_train = time.perf_counter()
            for step, batch in enumerate(tqdm(train_dataloader, desc = "Iteration")):
                batch = tuple(t.to(device) for t in batch)  # unpack and send each tensor to device
                input_ids, input_mask, segment_ids, label_ids = batch  # unpack the batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels = None).logits

                # TODO Why is it initialized here inside the batch loop? - bottleneck?
                loss_fct = CrossEntropyLoss()
                # calculate loss, make sure that the tensors have correct dimensionality
                loss = loss_fct(logits.view(-1, model.num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # calculate the backprogration with the loss
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # add the loss value to the logging variables
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # this is always executed if gradient_accumulation_steps = 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        raise ValueError("Using fp16 doublecheck everything!")
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        # lr_this_step = args.learning_rate * warmup_linear.get_lr(
                        #     global_step / num_train_optimization_steps, args.warmup_proportion)
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = lr_this_step
                    # update the weights and set optimizer to zero
                    optimizer.step()
                    # logger.debug(f'Current learning rate is: {optimizer.param_groups[0]["lr"]}')
                    linear_warmup_lr.step()
                    optimizer.zero_grad()
                    global_step += 1
            logger.info(f'Training loss: {tr_loss}')
            end_time_epoch_train = time.perf_counter()

            ############------------- VALIDATION ---------------#################
            # calculate validation loss
            model.eval()
            start_time_epoch_validate = time.perf_counter()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                      desc = "Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels = None).logits

                # create eval loss and other metric required by the task
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                # if len(preds) == 0:
                #     preds.append(logits.detach().cpu().numpy())
                # else:
                #     preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis = 0)

            eval_loss = eval_loss / nb_eval_steps
            #preds = preds[0]

            #preds = np.argmax(preds, axis = 1)
            #result = compute_metrics(task_name, preds, all_label_ids.numpy())

            end_time_epoch_validate = time.perf_counter()

            end_time_epoch = time.perf_counter()

            ############------------- LOGGING per epoch ---------------#################
            # add weights and gradients to tensorboard
            for parameter_name, values in model.named_parameters():
                if values.requires_grad is True:
                    writer_tb.add_histogram(parameter_name, values, i)
                    writer_tb.add_histogram(f'{parameter_name}.grad', values.grad, i)

            # add losses and evaluation metrics to tensorboard
            metric_dict = {'loss/training_loss': tr_loss,
                           'loss/validation_loss': eval_loss,
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
            tokenizer.save_vocabulary(
                os.path.join(DIRECTORY_FOR_SAVING_OR_LOADING, 'trained_model'))
            logger.info('Trained model saved to disk.')

        END_TRAINING = time.perf_counter()
        logger.info(
            f'Training and validation (without LP metrics) took {round((END_TRAINING - START_TRAINING) / 60, 2)} minutes in total.')

    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Save a trained model, configuration and tokenizer
    #     model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Only save the model it-self
    #
    #     # If we save using the predefined names, we can load using `from_pretrained`
    #     output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    #     output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    #
    #     # save model.bin, config.json and vocab.txt to disk
    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     model_to_save.config.to_json_file(output_config_file)
    #     tokenizer.save_vocabulary(args.output_dir)
    #
    #     # TODO Why is a model loaded here?
    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = BertForSequenceClassification.from_pretrained(args.output_dir,
    #                                                           num_labels = num_labels)
    #     tokenizer = BertTokenizer.from_pretrained(args.output_dir,
    #                                               do_lower_case = args.do_lower_case)
    # else:
    #     model = BertForSequenceClassification.from_pretrained(args.bert_model,
    #                                                           num_labels = num_labels)

        model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
            tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype = torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype = torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype = torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype = torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler = eval_sampler,
                                     batch_size = args.eval_batch_size)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir,
                                                              num_labels = num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir,
                                                  do_lower_case = args.do_lower_case)
        model.to(device)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                  desc = "Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels = None).logits

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            print(label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis = 0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        preds = np.argmax(preds, axis = 1)
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


    # IMPORTANT: calculation of link prediction metrics hapens here!
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):


        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, 'trained_model'),
                                                              num_labels = num_labels)
        tokenizer = BertTokenizer.from_pretrained(os.path.join(args.output_dir, 'trained_model'),
                                                  do_lower_case = args.do_lower_case)
        model.to(device)

        train_triples = processor.get_train_triples(args.data_dir)
        dev_triples = processor.get_dev_triples(args.data_dir)
        test_triples = processor.get_test_triples(args.data_dir)
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)

        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
            tokenizer)
        logger.info("***** Running Prediction *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype = torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype = torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype = torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype = torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler = eval_sampler,
                                     batch_size = args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                  desc = "Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels = None).logits

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis = 0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        #print(preds, preds.shape)

        all_label_ids = all_label_ids.numpy()

        preds = np.argmax(preds, axis = 1)

        result = compute_metrics(task_name, preds, all_label_ids)
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        #print("Triple classification acc is : ")
        #print(metrics.accuracy_score(all_label_ids, preds))

        # run link prediction
        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        top_ten_hit_count = 0

        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])
        '''
        file_prefix = str(args.data_dir[7:])
        f = open(file_prefix + '_ranks.txt','r')
        lines = f.readlines()
        for line in lines:
            temp = line.strip().split()
            rank1 = int(temp[0])
            ranks_left.append(rank1+1)
            print('left: ', rank1)
            ranks.append(rank1+1)
            if rank1 < 10:
                top_ten_hit_count += 1
            rank2 = int(temp[1])
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)
            print('right: ', rank2)
            print('mean rank until now: ', np.mean(ranks))
            if rank2 < 10:
                top_ten_hit_count += 1
            print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))                
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)
    
        '''
        test_triple_count = 0

        for test_triple in test_triples:
            logger.debug(
                f'Calculating rank for triple #{test_triple_count + 1} of {len(test_triples)}')
            start_time_test_triple = time.perf_counter()
            head = test_triple[0]
            relation = test_triple[1]
            tail = test_triple[2]
            logger.debug(f'Current test triple: {head, relation, tail}')

            head_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != head:
                    tmp_triple = [corrupt_ent, relation, tail]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        head_corrupt_list.append(tmp_triple)

            logger.info('######### Calculating rank head of current test triple #########')
            logger.debug(f'Length of head_corrupt list is: {len(head_corrupt_list)}')

            tmp_examples = processor._create_examples(head_corrupt_list, "test", args.data_dir)
            test_triple_as_text = f'{tmp_examples[0].text_a} | {tmp_examples[0].text_b} | {tmp_examples[0].text_c}'
            logger.debug(f'Current test triple as text is: {test_triple_as_text}')
            tmp_features = convert_examples_to_features(tmp_examples, label_list,
                                                        args.max_seq_length, tokenizer,
                                                        print_info = False)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype = torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype = torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features],
                                           dtype = torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype = torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for temp data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler = eval_sampler,
                                         batch_size = args.eval_batch_size)
            model.eval()

            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                      desc = "Testing"):

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels = None).logits
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)

                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis = 0)

            preds = preds[0]
            # get the dimension corresponding to current label 1
            # print(preds, preds.shape)
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            # print(rel_values, rel_values.shape)
            _, argsort1 = torch.sort(rel_values, descending = True)
            # print(max_values)
            # print(argsort1)
            argsort1 = argsort1.cpu().numpy()
            rank1 = np.where(argsort1 == 0)[0][0]
            logger.info(f'Rank head for current triple: {rank1}')
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            if rank1 < 10:
                top_ten_hit_count += 1

            tail_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != tail:
                    tmp_triple = [head, relation, corrupt_ent]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        tail_corrupt_list.append(tmp_triple)

            logger.info('######### Calculating rank tail of current test triple #########')
            logger.debug(f'Length of tail_corrupt list is: {len(tail_corrupt_list)}')

            tmp_examples = processor._create_examples(tail_corrupt_list, "test", args.data_dir)
            # print(len(tmp_examples))
            tmp_features = convert_examples_to_features(tmp_examples, label_list,
                                                        args.max_seq_length, tokenizer,
                                                        print_info = False)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype = torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype = torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features],
                                           dtype = torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype = torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for temp data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler = eval_sampler,
                                         batch_size = args.eval_batch_size)
            model.eval()
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                      desc = "Testing"):

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels = None).logits
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)

                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis = 0)

            preds = preds[0]
            # get the dimension corresponding to current label 1
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            _, argsort1 = torch.sort(rel_values, descending = True)
            argsort1 = argsort1.cpu().numpy()
            rank2 = np.where(argsort1 == 0)[0][0]
            logger.info(f'Rank tail for current triple: {rank2}')

            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)
            logger.info(f'mean rank until now:  {np.mean(ranks)}')

            if rank2 < 10:
                top_ten_hit_count += 1
            logger.info(f"hit@10 until now:  {top_ten_hit_count * 1.0 / len(ranks)}")

            file_prefix = "FB15k-237_test_" + str(args.train_batch_size) + "_" + str(
                args.learning_rate) + "_" + str(args.max_seq_length) + "_" + str(
                args.num_train_epochs)
            # file_prefix = str(args.data_dir[7:])
            f = open(file_prefix + '_ranks.txt', 'a')
            f.write(str(rank1) + '\t' + str(rank2) + '\n')
            f.close()
            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

            end_time_test_triple = time.perf_counter()
            runtime_for_this_triple = round((end_time_test_triple - start_time_test_triple), 2)
            logger.info(
                f'Rank calculation for current triple took {runtime_for_this_triple} seconds.')
            test_triple_count += 1

        ### Calculate all link prediction metrics after having gone through all test triples
        # Log hits @1, @3, @5 and @10
        for i in [0, 2, 4, 9]:
            logger.info(f'Hits head @{i + 1}: {np.mean(hits_left[i])}')
            logger.info(f'Hits tail @{i + 1}: {np.mean(hits_right[i])}')
            logger.info(f'Hits both @{i + 1}: {np.mean(hits[i])}')
        logger.info(f'Mean rank head: {np.mean(ranks_left)}')
        logger.info(f'Mean rank tail: {np.mean(ranks_right)}')
        logger.info(f'Mean rank both: {np.mean(ranks)}')
        logger.info(f'Mean reciprocal rank head: {np.mean(1. / np.array(ranks_left))}')
        logger.info(f'Mean reciprocal rank tail: {np.mean(1. / np.array(ranks_right))}')
        logger.info(f'Mean reciprocal rank both: {np.mean(1. / np.array(ranks))}')


if __name__ == "__main__":
    main()
