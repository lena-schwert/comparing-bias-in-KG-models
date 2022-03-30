import os
import torch

# TODO copy things here from utils.py

print(f'is the conda environment used. {os.system("echo $CONDA_DEFAULT_ENV")}')


print('######  CUDA ######')
print(f"CUDA is used: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f'CUDA detects {torch.cuda.device_count()} device(s).')
    print(f'CUDA device currently used: {torch.cuda.current_device()}')

# measure the RAM size of objects
import gc
import sys


def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    if memory_size >= 1024 * 1024:
        memory_size_MB = memory_size / (1024 * 1024)
        output = f'{memory_size} MB'
    output = f'{memory_size} Bytes'
    return output


# source: https://github.com/huggingface/transformers/issues/95

# for a given pytorch model, check whether parameters have been changed at all by training

import copy
from transformers import BertModel

bert = BertModel.from_pretrained('bert-base-uncased')
layer = bert.embeddings
frozen_parameters = {}

# Copy tensors
for name, p in layer.named_parameters():
    frozen_parameters[name] = copy.deepcopy(p.data) # Freeze in order to be able to compare later

# Do stuff ...

# Check if the value of the tensors have been updated
for name, p in layer.named_parameters():
    updated = (frozen_parameters[name] != p.data).any().cpu().detach().numpy()

    print(f"Layer '{name}' has been updated? {'yes' if updated else 'no'}")