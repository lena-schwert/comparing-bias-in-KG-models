# import from built-in modules
import os
import sys
import time
import gc

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 100)


# %% Preprocessing

# load dataframe from disk
tail_value_counts_df = pd.read_pickle('/home/lena/git/master_thesis_bias_in_NLP/exploration/relationship_counts/tail_value_counts_raw_10.11.2021.pkl')

# properly polish the dataframe
tail_value_counts_df = tail_value_counts_df.convert_dtypes()
tail_value_counts_df.reset_index(drop = True, inplace = True)
# change column names, get rid of hypens
tail_value_counts_df.columns = tail_value_counts_df.columns.str.replace('-', '_')

# %% after retrieving all counts for the datasets:
# map from Q-IDs to English Wikidata labels for human readability
# USE Wikidatasets-Humans entities.tsv
dataset_folder = '/home/lena/git/master_thesis_bias_in_NLP/data/Wikidatasets_humans/'
Q_IDs_to_labels = pd.read_csv(os.path.join(dataset_folder, 'entities.tsv'),
                              sep = '\t', skiprows = 1,
                              names = ['Wikidatasets_ID', 'wikidata_qid', 'wikidata_label'])
Q_IDs_to_labels.drop('Wikidatasets_ID', axis = 1, inplace = True)

# iterate through each single row?
for row in tail_value_counts_df.itertuples():
    print(row)
    Q_IDs_to_labels.query(f'wikidata_qid=={row.tail_entity_Q_ID}')

    pass





# transform counts into percentages




# total rows: 894648
# counts >= 10 are 119,734 rows
# counts == 1 are 473870
tail_value_counts_df.query('count>=10000')


# filter out low percentages
# make histogram plot to see where to cut off
counts_larger_equal_10 = tail_value_counts_df.query('count>=10')

# this takes too long
#sns.histplot(data = counts_larger_equal_10)

ax = counts_larger_equal_10.plot.hist(by = 'dataset_name')
plt.show()

plot_only_count = counts_larger_equal_10['count'].plot.hist()


sns.histplot(data = counts_larger_equal_10,
             x = 'count')

# TODO later on, cretae OTHER category



# %% Create plots of tail value counts

# Which count values occur?



