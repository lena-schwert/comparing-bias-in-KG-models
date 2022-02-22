
## *** Work in Progress - Project is still on-going ***

## Structure of this project

The logic of the project structure is mostly based on: https://drivendata.github.io/cookiecutter-data-science/#directory-structure

The `src` folder contains all python files except those used in exploration, as these come with no guarantees for usability.


## Setting file paths

All scripts are using the function `set_base_path_based_on_host()` in
`utils.py` to set the **base path for saving and loading files**.

Make sure that you adapt the path in this function to your project.



## References to related projects

This project is built on previous literature and therefore uses 
code from repositories published by the respective authors.

These are:

https://github.com/yao8839836/kg-bert

https://github.com/mianzg/kgbiasdetec

https://github.com/russabiswas/Contextual_Language_Models_for_KGC

https://github.com/merialdo/research.lpbias
