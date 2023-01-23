#!/bin/bash

echo "Script started"
date

set -x # This makes bash print all commands to stdout
set -e # Exit immediately if a command exits with a non-zero status.

################## EVALUATE the IB MODEL on VAL and

MODEL_PATH="/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_and_LM/SimKGC/12.05.2022_19:34_wiki5m_trans_first_try_SimKGC_4xa6k5/model_best.mdl"
TASK="wiki5m_trans"
DIR="/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis"
echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}"
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_PATH=$1
    shift
fi

neighbor_weight=0.05


TEST_PATH="${DATA_DIR}/validation_data_subset_0.05.tsv.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TEST_PATH=$1
    shift
fi
python3 -u eval_wiki5m_trans.py \
--task "${TASK}" \
--is-test \
--eval-model-path "${MODEL_PATH}" \
--neighbor-weight "${neighbor_weight}" \
--train-path "${DATA_DIR}/training_data_subset_0.9.tsv.json" \
--rerank-n-hop 2 \
--valid-path "${TEST_PATH}" "$@" | tee log_evaluate_valset_SimKGC_IB_21.5.2022.txt

################## EVALUATE the IB + SN + PB model on VAL and TEST

MODEL_PATH="/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_and_LM/SimKGC/14.05.2022_00:21_wiki5m_trans_train_SimKGC_IBSNPB_4xa6k5/model_best.mdl"
TASK="wiki5m_trans"
DIR="/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis"
echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}"
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_PATH=$1
    shift
fi

neighbor_weight=0.05

TEST_PATH="${DATA_DIR}/validation_data_subset_0.05.tsv.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TEST_PATH=$1
    shift
fi
python3 -u eval_wiki5m_trans.py \
--task "${TASK}" \
--is-test \
--eval-model-path "${MODEL_PATH}" \
--neighbor-weight "${neighbor_weight}" \
--train-path "${DATA_DIR}/training_data_subset_0.9.tsv.json" \
--rerank-n-hop 2 \
--valid-path "${TEST_PATH}" "$@" | tee log_evaluate_valset_SimKGC_IBSNPB_21.5.2022.txt

TEST_PATH="${DATA_DIR}/test_data_subset_0.05.tsv.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TEST_PATH=$1
    shift
fi
python3 -u eval_wiki5m_trans.py \
--task "${TASK}" \
--is-test \
--eval-model-path "${MODEL_PATH}" \
--neighbor-weight "${neighbor_weight}" \
--train-path "${DATA_DIR}/training_data_subset_0.9.tsv.json" \
--rerank-n-hop 2 \
--valid-path "${TEST_PATH}" "$@" | tee log_evaluate_testset_SimKGC_IBSNPB_21.5.2022.txt



echo "Script ended"
date
echo "Host is"
echo hostname
