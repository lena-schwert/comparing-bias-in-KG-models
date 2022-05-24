#!/bin/bash


MODEL=RotatE
DIM=512
NEGATIVE_SAMPLES=32
BATCH_SIZE=1024
LEARNING_RATE=0.01

# train the model in multiple steps to save models at different states:
EPOCH=100

python experiments_KG_only.py --kge $MODEL -n "final_model_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES


