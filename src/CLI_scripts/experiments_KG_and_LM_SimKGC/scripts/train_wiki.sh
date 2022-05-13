#!/usr/bin/env bash

echo "Script started"
date

set -x  # This makes bash print all commands to stdout
set -e  # Exit immediately if a command exits with a non-zero status.

TASK="wiki5m_trans"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

EXPERIMENT_NAME=first_try_SimKGC_4xa6k5

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
DIR="/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/results/KG_and_LM/SimKGC/$(date +%d.%m.%Y_%H:%M)_${TASK}_${EXPERIMENT_NAME}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--train-path "${DATA_DIR}/training_data_subset_0.9.tsv.json" \
--valid-path "${DATA_DIR}/validation_data_subset_0.05.tsv.json" \
--task "${TASK}" \
--batch-size 1024 \
--epochs 1 \
--lr 3e-5 \
--pre-batch 0 \
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
--max-to-keep 10  "$@"

echo "Script finished."
date
