#!/bin/bash
#SBATCH -A demelo
#SBATCH -p sorcery
#SBATCH -C GPU_SKU:A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=35
#SBATCH --mem=30GB
#SBATCH -t 58:0:0

# try different gradient accumulation step sizes
python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr5e-5_ga2 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 5e-5  --gradient_accumulation_steps 2 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr5e-5_ga4 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 5e-5  --gradient_accumulation_steps 4 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr5e-5_ga8 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 5e-5  --gradient_accumulation_steps 8 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr5e-5_ga16 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 5e-5  --gradient_accumulation_steps 16 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs32_lr5e-5_ga32 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 32 --learning_rate 5e-5  --gradient_accumulation_steps 32 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs64_lr5e-5_ga16 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 64 --learning_rate 5e-5  --gradient_accumulation_steps 32 

python experiments_KG_and_LM.py --name HPO_FB15k237_bs64_lr5e-5_ga4 --do_train --data_dir /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/code_from_other_papers/Yao_KG_BERT/data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --eval_batch_size 1500 --num_train_epochs 5 --train_batch_size 64 --learning_rate 5e-5  --gradient_accumulation_steps 4
