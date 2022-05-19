import os
from src.utils import set_base_path_based_on_host


ROOT = set_base_path_based_on_host()
DATA_PATH = os.path.join(ROOT, "data/interim")
BIAS_DATA_PATH = os.path.join(ROOT, "results/bias_measurement/data_bias/Rossi_bias_types")
RESULTS_PATH = os.path.join(ROOT, "results/bias_measurement/data_bias/comparative_analysis_results")

# make sure that these folders exist
for path in [DATA_PATH, BIAS_DATA_PATH, RESULTS_PATH]:
    assert os.path.isdir(path), f'Folder {path} does not exist!'

# model names
ANALOGY = "ANALOGY"
ANYBURL = "AnyBURL-RE"
CAPSE = "CapsE"
COMPLEX = "ComplEx"
CONVE = "ConvE"
CONVKB = "ConvKB"
CONVR = "ConvR"
CROSSE = "CrossE"
DISTMULT = "DistMult"
HAKE = "HAKE"
HOLE = "HolE"
INTERACTE = "InteractE"
ROTATE = "RotatE"
RSN = "RSN"
SIMPLE = "SimplE"
STRANSE = "STransE"
TORUSE = "TorusE"
TRANSE = "TransE"
TUCKER = "TuckER"

ALL_MODEL_NAMES = [DISTMULT, COMPLEX, ANALOGY, SIMPLE, HOLE, TUCKER,
                   TRANSE, STRANSE, CROSSE, TORUSE, ROTATE, HAKE,
                   CONVE, CONVKB, CONVR, INTERACTE, CAPSE, RSN,
                   ANYBURL]

SELECTED_MODEL_NAMES = [TRANSE]

# dataset names
FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10 = "YAGO3-10"
HUMANWIKIDATA5M = "HumanWikidata5M"
ALL_DATASET_NAMES = [FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, HUMANWIKIDATA5M]

SELECTED_DATASET_NAMES = [HUMANWIKIDATA5M]
