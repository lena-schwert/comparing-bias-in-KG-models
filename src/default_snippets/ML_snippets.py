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