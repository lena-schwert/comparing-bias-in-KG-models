#!/usr/bin/env bash

echo "Preprocessing has started."
date

set -x # This makes bash print all commands to stdout
set -e # Exit immediately if a command exits with a non-zero status.

TASK="wiki5m_trans"
if [[ $# -ge 1 ]]; then
    TASK=$1
    shift
fi

#DIR="/scratch1/lschwertmann/master_thesis_synced"
DIR="/home/lena/git/master_thesis_bias_in_NLP/"
echo "working directory: ${DIR}"

python3 -u preprocess.py \
--task "${TASK}" \
--train-path "${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}/training_data_subset_0.9_rs42_06_05_2022_15:11.tsv" \
--valid-path "${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}/validation_data_subset_0.05_rs42_06_05_2022_15:11.tsv" \
--test-path "${DIR}/data/processed/files_per_model/KG_and_LM_SimKGC_subset_HumanW5M/${TASK}/test_data_subset_0.05_rs42_06_05_2022_15:11.tsv"

echo "Preprocessing has ended."
date