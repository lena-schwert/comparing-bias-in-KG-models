#!/bin/bash
# ssh to node, cd to folder, start tmux,


# starting out from ssh Terminal started by Pycharm


# starting out from LOGIN node
source cd_to_scratch
tmux a -t DGX
salloc -A demelo -p sorcery -w dgxa100-01 --gpus=1 -t 3:0:0
conda activate master-thesis_cuda113

