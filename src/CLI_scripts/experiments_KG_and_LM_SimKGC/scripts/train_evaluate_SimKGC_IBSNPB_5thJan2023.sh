#!/bin/bash

################## TRAIN the IB + SN + PB model

echo "Training script started"
date

export CUDA_VISIBLE_DEVICES=5,6 && echo "The IDs of the CUDA devices used are: " && echo $CUDA_VISIBLE_DEVICES

START_TIME=$(date +%d.%m.%Y_%H:%M)

set -x  # This makes bash print all commands to stdout
set -e  # Exit immediately if a command exits with a non-zero status.

TASK="wiki5m_trans"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

EXPERIMENT_NAME=SimKGC_IBSNPB_4xAIISGPUs_mascul_fem_descr
DIR="/scratch1/lschwertmann/master_thesis_synced"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/results/KG_and_LM/SimKGC/${START_TIME}_${TASK}_${EXPERIMENT_NAME}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--train-path "${DATA_DIR}/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv.json" \
--valid-path "${DATA_DIR}/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv.json" \
--task "${TASK}" \
--batch-size 1024 \
--epochs 1 \
--lr 3e-5 \
--pre-batch 2 \
--pre-batch-weight 0.5 \
--use-self-negative \
--max-num-tokens 50 \
--pooling mean \
--additive-margin 0.02 \
--warmup 400 \
--grad-clip 10.0 \
--dropout 0.1 \
--use-amp \
--finetune-t \
--weight-decay 1e-4 \
--lr-scheduler 'linear'  \
--eval-every-n-step 10000 \
--workers 3 \
--print-freq 20 \
--seed 42 \
--max-to-keep 10  "$@"  | tee /scratch1/lschwertmann/master_thesis_synced/results/KG_and_LM/SimKGC/${START_TIME}_${TASK}_${EXPERIMENT_NAME}/log_train_${EXPERIMENT_NAME}_${START_TIME}.txt

echo "Script finished."
date


################## EVALUATE the IB + SN + PB model on the validation set

echo "Script started"
date

#set -x # This makes bash print all commands to stdout
#set -e # Exit immediately if a command exits with a non-zero status.

MODEL_PATH="${DIR}/results/KG_and_LM/SimKGC/${START_TIME}_${TASK}_${EXPERIMENT_NAME}/model_best.mdl"
TASK="wiki5m_trans"

echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}"
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_PATH=$1
    shift
fi

TEST_PATH="${DATA_DIR}/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TEST_PATH=$1
    shift
fi

neighbor_weight=0.05

python3 -u eval_wiki5m_trans.py \
--task "${TASK}" \
--is-test \
--eval-model-path "${MODEL_PATH}" \
--neighbor-weight "${neighbor_weight}" \
--train-path "${DATA_DIR}/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv.json" \
--rerank-n-hop 2 \
--valid-path "${TEST_PATH}" "$@" | tee /scratch1/lschwertmann/master_thesis_synced/results/KG_and_LM/SimKGC/${START_TIME}_${TASK}_${EXPERIMENT_NAME}/log_val_${EXPERIMENT_NAME}_${START_TIME}.txt






