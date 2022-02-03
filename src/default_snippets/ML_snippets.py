import os
import torch

# TODO copy things here from utils.py

print(f'is the conda environment used. {os.system("echo $CONDA_DEFAULT_ENV")}')


print('######  CUDA ######')
print(f"CUDA is used: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f'CUDA detects {torch.cuda.device_count()} device(s).')
    print(f'CUDA device currently used: {torch.cuda.current_device()}')