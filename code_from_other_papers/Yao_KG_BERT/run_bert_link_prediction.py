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

import argparse
import csv
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from sklearn import metrics

# ORIGINAL IMPORTS FROM KG BERT CODE
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

# IMPORTS FROM BISWAS PAPER (KG-GPT-2)
# from transformers import (set_seed, TrainingArguments, Trainer, GPT2Config, GPT2Tokenizer,
# AdamW, get_linear_schedule_with_warmup, GPT2ForSequenceClassification)

# my imports using up-to-date transformers module and BERT
# missing: WEIGHTS_NAME, CONFIG_NAME
# imports that are not used in the original code: BertConfig
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.file_utils import TRANSFORMERS_CACHE
from transformers import AdamW, \
    get_linear_schedule_with_warmup  # instead of BertAdam, WarmupLinearSchedule

START_TIME = datetime.now().strftime("%d.%m.%Y_%H:%M")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# torch.backends.cudnn.deterministic = True

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
                    raise SystemError('sys.version_info[0] == 2, I do not know what that means! Check code')
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(# set type "train" here for the _create_examples function
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
        logger.debug('Reading train.tsv...')
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        logger.debug('Reading dev.tsv...')
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        logger.debug('Reading test.tsv...')
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        # TODO access to file entity2text.txt
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                # Why is this check necessary?
                if len(temp) == 2:
                    # select the second entry = entity as text
                    end = temp[1]  # .find(',')
                    ent2text[temp[0]] = temp[1]  # [:end]

        # only execute this code for FB15K and FB15K-237 datasets
        if data_dir.find("FB15") != -1:
            logger.debug('Using longer entity descriptions (entity2textlong.txt) for Freebase datasets.')
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

        logger.info('Creating examples ')
        for (i, line) in enumerate(tqdm(lines)):

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

    logger.debug('Creating features, i.e. vectors from the previously created examples (which are text instead of entity/relation IDs).')
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # TODO remove this when no longer needed, this is a lot!
        logger.debug()

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
        if ex_index < 5 and print_info:
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


# TODO custom code by Lena: measure the RAM size of objects
import gc


def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    if memory_size >= 1024 * 1024:
        memory_size_MB = memory_size / (1024 * 1024)
        output = f'{memory_size} MB'
    output = f'{memory_size} Bytes'
    return output


def main():
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--task_name", default = 'kg', type = str, required = True,
                        help = "The name of the task to train.")
    parser.add_argument("--output_dir",
                        default = '/home/lena/git/master_thesis_bias_in_NLP/code_from_other_papers/Yao_KG_BERT/output_FB15k237',
                        type = str, required = True,
                        help = "The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
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

    # set the correct CUDA device, check for number of devices
    if args.local_rank == -1 or args.no_cuda:
        # important: if no_cuda is enabled, use CPU even though GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend = 'nccl')

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # configure the logging
    # important: custom logging by Lena to stdout and file
    format = '%(asctime)s - %(levelname)s - %(filename)s/%(funcName)s: %(message)s'
    logging.basicConfig(format = format,
                        level = logging.DEBUG if args.local_rank in [-1, 0] else logging.WARN,
                        datefmt = "%d.%m.%Y %H:%M:%S", handlers = [
            logging.FileHandler(f'log_{os.path.split(args.data_dir)[1]}_{START_TIME}', mode = 'a'),
            logging.StreamHandler(sys.stdout)])

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu,
            bool(args.local_rank != -1), args.fp16))

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

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # create a KGProcessor + task name
    processors = {"kg": KGProcessor, }

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # create a KGProcessor object
    processor = processors[task_name]()

    # access to entities.txt file to create a list object
    # important access to file entities.txt
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)
    entity_list = processor.get_entities(args.data_dir)

    # important added custom logging
    logger.info(f'There are {len(entity_list)} entities in the dataset.')
    logger.info(f'Entity list takes {actualsize(entity_list)} of RAM.')

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = args.do_lower_case)

    # get training examples + create NEGATIVE/CORRUPT triples for each of it

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        # important: this call takes a long time!
        logger.info('Creating examples from the training split of the dataset.')
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model, download BERT for sequence classification
    # create cache folder so the model is only downloaded when the script is run for the first time on the machine
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(TRANSFORMERS_CACHE),
                                                                   'distributed_{}'.format(
                                                                       args.local_rank))

    # parameters will be loaded as float 32
    model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir = cache_dir,
                                                          num_labels = num_labels)

    # if specified, cast the model to float 16 datatype
    if args.fp16:
        model.half()

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

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # show all parameter names
    # for name, param in model.named_parameters():
    #    print(name)

    # add all parameters except decay?
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # TODO unclear: set weight decay to zero?
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    # use other optimizers if using float16 data type
    if args.fp16:
        logger.debug('Setting optimizer for float 16 model.')
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters, lr = args.learning_rate,
                              bias_correction = False, max_grad_norm = 1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale = True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = args.loss_scale)
        # warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
        #                                      t_total=num_train_optimization_steps)
        warmup_linear = get_linear_schedule_with_warmup()

    else:
        # optimizer = BertAdam(optimizer_grouped_parameters,
        #                      lr=args.learning_rate,
        #                      warmup=args.warmup_proportion,
        #                      t_total=num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate,
                          correct_bias = False)  # Important: To reproduce BertAdam specific behavior set correct_bias=False

    global_step = 0  # ?
    nb_tr_steps = 0
    tr_loss = 0  # accumulate training loss?

    # DO THE TRAINING!!!
    if args.do_train:

        train_features = convert_examples_to_features(train_examples, label_list,
            args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

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

        model.train()  # set model to train mode

        for _ in trange(tqdm(int(args.num_train_epochs)), desc = "Epoch"):
            # note that global_step is not set to zero
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc = "Iteration")):
                batch = tuple(t.to(device) for t in batch)  # unpack and send each tensor to device
                input_ids, input_mask, segment_ids, label_ids = batch  # unpack the batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels = None)
                # print(logits, logits.shape)

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
                    # TODO decide whether to add gradient clipping (part of pytorch_pretrained migration)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # add the loss value to the logging variables
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # this is always executed if gradient_accumulation_steps = 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step / num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    # update the weights and set optimizer to zero
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            print("Training loss: ", tr_loss, nb_tr_examples)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        # TODO output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # TODOoutput_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        # save model.bin, config.json and vocab.txt to disk
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # TODO Why is a model loaded here?
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir,
                                                              num_labels = num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir,
                                                  do_lower_case = args.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                              num_labels = num_labels)

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
                logits = model(input_ids, segment_ids, input_mask, labels = None)

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

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

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
                                                                  desc = "Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels = None)

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
        print(preds, preds.shape)

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
        print("Triple classification acc is : ")
        print(metrics.accuracy_score(all_label_ids, preds))

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
        for test_triple in test_triples:
            head = test_triple[0]
            relation = test_triple[1]
            tail = test_triple[2]
            # print(test_triple, head, relation, tail)

            head_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != head:
                    tmp_triple = [corrupt_ent, relation, tail]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        head_corrupt_list.append(tmp_triple)

            tmp_examples = processor._create_examples(head_corrupt_list, "test", args.data_dir)
            print(len(tmp_examples))
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
                    logits = model(input_ids, segment_ids, input_mask, labels = None)
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
            print('left: ', rank1)
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
                    logits = model(input_ids, segment_ids, input_mask, labels = None)
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
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)
            print('right: ', rank2)
            print('mean rank until now: ', np.mean(ranks))
            if rank2 < 10:
                top_ten_hit_count += 1
            print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

            file_prefix = str(args.data_dir[7:]) + "_" + str(args.train_batch_size) + "_" + str(
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

        for i in [0, 2, 9]:
            logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
            logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
            logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
        logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))


if __name__ == "__main__":
    main()
