#!/bin/bash


MODEL=DistMult
DIM=512
NEGATIVE_SAMPLES=32
BATCH_SIZE=1024
LEARNING_RATE=0.001

# train the model in multiple steps to save models at different states:

EPOCH=200
python experiments_KG_only.py --load-checkpoint --kge $MODEL -n "23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_100ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

cp -r /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/final/23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_100ep /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/final/23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_after_200ep

EPOCH=300
python experiments_KG_only.py --load-checkpoint --kge $MODEL -n "23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_100ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

cp -r /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/final/23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_100ep /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/final/23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_after_300ep

EPOCH=400
python experiments_KG_only.py --load-checkpoint --kge $MODEL -n "23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_100ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

cp -r /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/final/23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_100ep /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/final/23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_after_400ep

EPOCH=500
python experiments_KG_only.py --load-checkpoint --kge $MODEL -n "23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_100ep" -e $EPOCH -bs $BATCH_SIZE -lr $LEARNING_RATE --dim $DIM -ns $NEGATIVE_SAMPLES

cp -r /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/final/23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_100ep /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/results/KG_only/final/23.05.2022_20:06_final_model_DistMult_512dim_32ns_1024bs_0.001lr_after_500ep
