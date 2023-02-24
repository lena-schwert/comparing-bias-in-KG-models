
## EACL Findings 2023: Model-Agnostic Bias Measurement in Link Prediction

This repository contains the code for implementing all experiments described in our paper. It also links to the data files for *HumanW5M-3mil* a dataset that we create based on the [Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m) benchmark dataset.

Our dataset is an enhanced subset containing 3 milion facts about humans. For each entity in the dataset, we provide *descriptions* that correspond to the first section of the respective English Wikipedia article, as well as *labels* that correspond to the English Wikidata label.


## Setting file paths

All scripts are using the function `set_base_path_based_on_host()` in
`utils.py` to set the **base path for saving and loading files**.

Make sure that you adapt the path in this function to your project.


## Structure of this project

`data`


|-- The processed data files used for training all models are contained in the folder `data/processed`.

`src`: contains all python files and command line scripts.







## References to related projects

This project is built on previous literature and therefore partly uses 
code from repositories published by the respective authors.

These are:

https://github.com/mianzg/kgbiasdetec

https://github.com/intfloat/SimKGC


