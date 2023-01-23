#!/bin/bash

echo "Script started"
date

set -x # This makes bash print all commands to stdout
set -e # Exit immediately if a command exits with a non-zero status.

############# set CUDA devices for AIIS GPUs!

export CUDA_VISIBLE_DEVICES=0 && echo "The IDs of the CUDA devices used are: " && echo $CUDA_VISIBLE_DEVICES

################## EVALUATE the IB on TEST

TASK="wiki5m_trans"
DIR=/scratch1/lschwertmann/master_thesis_synced
RESULT_FOLDER=results/KG_and_LM/SimKGC
EXPERIMENT_NAME="10.01.2023_13:02_wiki5m_trans_SimKGC_IB_2xAIISGPUs_mascul_fem_descr"
MODEL_PATH="${DIR}/${RESULT_FOLDER}/${EXPERIMENT_NAME}/model_best.mdl"
echo "working directory: ${DIR}"

START_TIME=$(date +%d.%m.%Y_%H:%M)


if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}"
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_PATH=$1
    shift
fi

neighbor_weight=0.05

##################### evaluate on TEST

TEST_PATH="${DATA_DIR}/test_data_subset_0.05_rs42_06_05_2022_15:11.tsv.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TEST_PATH=$1
    shift
fi
python3 -u eval_wiki5m_trans.py \
--task "${TASK}" \
--is-test \
--eval-model-path "${MODEL_PATH}" \
--neighbor-weight "${neighbor_weight}" \
--train-path "${DATA_DIR}/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv.json" \
--rerank-n-hop 2 \
--valid-path "${TEST_PATH}" "$@" | tee ${DIR}/${RESULT_FOLDER}/${EXPERIMENT_NAME}/log_evaluate_testset_${EXPERIMENT_NAME}_${START_TIME}.txt

echo "Script ended"
date
echo "Host is"
echo hostname
