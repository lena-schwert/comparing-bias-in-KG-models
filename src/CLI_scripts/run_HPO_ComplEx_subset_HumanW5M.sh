#!/bin/bash
#SBATCH -A demelo
#SBATCH -p sorcery
#SBATCH -w a6k5-01
#SBATCH --reservation=res_schwertmann_1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=40GB
#SBATCH -t 5-00:00:00
#SBATCH --mail-user=lena.schwertmann@guest.hpi.de
#SBATCH --mail-type=ALL
#SBATCH -e /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/HPO/slurm-%j_ComplEx.txt
#SBATCH -o /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/HPO/slurm-%j_ComplEx.txt

#set -x

MODEL=ComplEx
DIM=512
EPOCH=20
NEGATIVE_SAMPLES=32

BATCH_SIZE=1024
LEARNING_RATE=0.001
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=1024
LEARNING_RATE=0.01
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=1024
LEARNING_RATE=0.1
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=512
LEARNING_RATE=0.001
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=512
LEARNING_RATE=0.01
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=512
LEARNING_RATE=0.1
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

DIM=100
BATCH_SIZE=1024
LEARNING_RATE=0.001
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=1024
LEARNING_RATE=0.01
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=1024
LEARNING_RATE=0.1
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=512
LEARNING_RATE=0.001
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=512
LEARNING_RATE=0.01
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=512
LEARNING_RATE=0.1
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

DIM=512
NEGATIVE_SAMPLES=64
BATCH_SIZE=1024
LEARNING_RATE=0.001
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

BATCH_SIZE=1024
LEARNING_RATE=0.01
python experiments_KG_only.py --kge $MODEL -n "HPO_${MODEL}_${DIM}dim_${NEGATIVE_SAMPLES}ns_${BATCH_SIZE}bs_${LEARNING_RATE}lr_${EPOCH}ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES
