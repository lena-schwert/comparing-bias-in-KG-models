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

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
DIR="/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/results/KG_and_LM/SimKGC/${TASK}_$(date +%F-%H%M.%S)"
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
--batch-size 1024 \   # like paper
--epochs 1 \   # like paper
--lr 3e-5 \   # like paper
--pre-batch 0 \    # set this to 2 for PB type of negative samples
#--pre-batch-weight 0.5 \    # the weight for logits from pre-batch negatives
#--use-self-negative \   # add this for SB type of negative samples
--max-num-tokens 50 \      # default, like paper
--pooling mean \    # like paper
--additive-margin 0.02 \    # like paper, for InfoNCEloss
--warmup 400 \      # default, like paper, warmup steps
--grad-clip 10.0 \      # default, like paper,   gradient clipping
--dropout 0.1 \    # # default, like paper, dropout on final linear layer
--use-amp \   # use amp if available
--finetune-t \   # if added, make temperature as a trainable parameter or not
# alternatively: --t 0.05 \ # default
--weight-decay 1e-4 \   # default, like paper
--lr-scheduler 'linear'  \      # default, not mentioned in paper
--eval-every-n-step 10000 \    # default
--workers 3 \
--print-freq 20 \
--seed 42 \  # if not None, enables cudnn.deterministic = True in config.py
# max number of checkpoints to keep
--max-to-keep 10  "$@"




echo "Script finished."
date
