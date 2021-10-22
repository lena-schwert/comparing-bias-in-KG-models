# 1.9.2021

# libraries:
# libKG, pykeen

# tasks:
# extract a list of all entities in Wikidata
# extract some data from Wikidata and calculate embeddings from it
# select a specific property and extract some 100s of entities that have this property

# %% General utilities
import time


# %% Trying out some tools

import wikirepo
from wikirepo.data import data_utils, lctn_utils, time_utils, wd_utils


# from example: https://colab.research.google.com/github/andrewtavis/poli-sci-kit/blob/main/examples/global_parliament.ipynb#scrollTo=ttd7vTqXHQcD
ents_dict = wd_utils.EntitiesDict()
timespan = None
interval = None
depth = 0

locations = ['Germany', 'Norway', 'France']

# population information about the countries specified
results_df = wikirepo.data.query(ents_dict=ents_dict,
                         depth=depth, locations=locations,
                         timespan=timespan, interval=interval,
                         demographic_props='population',
                         economic_props=None,
                         electoral_poll_props=None,
                         electoral_result_props=None,
                         geographic_props='continent',
                         institutional_props=['fh_category', 'org_membership'],
                         political_props=None,
                         misc_props=None,
                         verbose=False) # False for web display

# another variant from here: https://github.com/andrewtavis/wikirepo/blob/main/src/wikirepo/data/demographic/ethnic_div.py
# ethnic diversity:

pid = "P172"
sub_pid = "P1107"
col_name = None  # col_name is None for no data col
col_prefix = "eth"  # columns will be generated and prefixed from values
ignore_char = ""
span = False

from wikirepo.data import data_utils
df, ents_dict = data_utils.query_wd_prop(dir_name = None, ents_dict = ents_dict,
    locations = locations, depth = depth, timespan = timespan, interval = interval, pid = pid,
    sub_pid = sub_pid, col_name = col_name, col_prefix = col_prefix, ignore_char = ignore_char,
    span = span, )

# %% WikidataIntegrator

from wikidataintegrator import wdi_core
my_first_wikidata_item = wdi_core.WDItemEngine(wd_item_id='Q5')
my_first_wikidata_item.get_wd_json_representation()
