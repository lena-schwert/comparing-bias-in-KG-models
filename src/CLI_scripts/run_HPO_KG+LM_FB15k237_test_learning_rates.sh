#!/bin/bash
#SBATCH -A demelo
#SBATCH -p sorcery
#SBATCH -C GPU_SKU:A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=30GB
#SBATCH -t 40:0:0

# try different learning rates
python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr2e-5_ga1 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 2e-5  --gradient_accumulation_steps 1 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr1e-4_ga1 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 1e-4  --gradient_accumulation_steps 1 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr1e-3_ga1 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 1e-3  --gradient_accumulation_steps 1 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr1e-2_ga1 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 1e-2  --gradient_accumulation_steps 1 
