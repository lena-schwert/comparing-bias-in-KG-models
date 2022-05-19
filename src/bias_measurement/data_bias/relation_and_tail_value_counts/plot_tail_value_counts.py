# import from built-in modules
import os
import pickle
import sys
import time
import gc
import socket

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 100)

# %% Load dataframe

# set base path
if socket.gethostname() == 'Schlepptop':
    base_dir = '/home/lena/git/master_thesis_bias_in_NLP/'
# covers all CPU nodes
elif 'node' in socket.gethostname():
    base_dir = '/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis'


# load dataframe from disk
tail_value_counts_df = pd.read_pickle(os.path.join(base_dir,
                                                   'exploration/relationship_counts/tail_value_counts_all_11.11.2021.pkl'))

# %% Inspect missing Wikidata labels

# check with isna() whether all Q-IDs have a label now
NA_filtering_mask = pd.isna(tail_value_counts_df['tail_entity_label'])
tail_value_counts_df_NA_labels = tail_value_counts_df[NA_filtering_mask]
count_unique_NA_Q_IDs = tail_value_counts_df_NA_labels['tail_entity_Q_ID'].unique()
print(f'There are {len(count_unique_NA_Q_IDs)} unique Q-IDs that do not have a label!')
# overall there are 17695 unique entities that don't have a label!
print(f'These IDs have {len(tail_value_counts_df_NA_labels.iloc[:,0])} occurrences across all datasets and relations.')

# How relevant is it? Do any of these NA entities have high counts?
tail_value_counts_df_NA_labels.query('count>=1000')
# 30 rows have counts >= 1000
tail_value_counts_df_NA_labels['count'].max()  # largest count is 4,213

tail_value_counts_df_NA_labels.query('count<=5')  # thereof 19,094 with very low count

len_NA_count_1 = len(tail_value_counts_df_NA_labels.query('count==1')['tail_entity_Q_ID'].unique())
len_NA_count_smaller_5 = len(tail_value_counts_df_NA_labels.query('count<=5')['tail_entity_Q_ID'].unique())

print(f'Of these occurrences, {len_NA_count_1} have a count == 1, so they are not of interest.')
print(f' Overall {len_NA_count_smaller_5} have a count <=5 so they also do not matter')
print('Only 30 rows have a count larger than 1000.')

# %% simple histogram plot of all tail value counts across all datasets

# Which count values occur?
# Serves as a decision threshold for the accumulative OTHER category

# Inspect counts manually using query()

# total rows: 894648
# counts >= 10 are 119,734 rows
# counts == 1 are 473870 rows
# counts <=3 are 657,956 rows
tail_value_counts_df.query('count>=10000')

# make histogram plot to see where to cut off
# plot the pandas series directly with pandas
type(tail_value_counts_df['count'])

tail_value_counts_df.hist(by = 'count')
# max: 4470698 counts
# min: 1

series = tail_value_counts_df['count']

series.hist(by = 'count')

# try again with seaborn
sns.histplot(x = series)

sns.histplot(data = tail_value_counts_df, x = 'count')

# make it smaller
# the full dataframe can't be plotted
tail_value_counts_df_counts_more_than_100 = tail_value_counts_df.query('count>=100')  # 21,606 rows
tail_value_counts_df_counts_more_than_1000 = tail_value_counts_df.query('count>=1000')  # 3,087 rows

# 21k rows
# this fully uses one CPU
# takes several minutes to be executed
# executes faster when specifying binwidth
sns.histplot(data = tail_value_counts_df_counts_more_than_1000, x = 'count', binwidth = 10)

# try casting series to numpy array
# try to plot all values with binwidth 1000
count_np_array = tail_value_counts_df['count'].to_numpy()
count_np_array_counts_more_than_1000 = tail_value_counts_df_counts_more_than_1000[
    'count'].to_numpy()
sns.histplot(data = count_np_array_counts_more_than_1000)

# use numpy directly for plotting the histogram
np.histogram(count_np_array_counts_more_than_1000)

# use pandas value counts for plotting
value_counts_pandas = tail_value_counts_df['count'].value_counts()
value_counts_pandas.hist(bins = 10)
value_counts_pandas_counts_more_than_1000 = tail_value_counts_df_counts_more_than_1000[
    'count'].value_counts()

# seaborn: use displot() or histplot()
sns.displot(value_counts_pandas_counts_more_than_1000, kind = 'hist', rug = True)

value_counts_pandas_counts_more_than_1000.sort_index(inplace = True)
sns.barplot(y = value_counts_pandas_counts_more_than_1000,
            x = value_counts_pandas_counts_more_than_1000.index)
plt.show()

plt.show()

# sanity check: can I plot it for a really low amount of items?


# %% simple histogram plot of all tail value counts across all datasets

threshold_for_minimum_count = 10000

tail_value_counts_df_filtered_for_counts = tail_value_counts_df.query(
    f'count>={threshold_for_minimum_count}')
# 2 --> 420,741 rows
# 3 --> 297,323 rows
# 5 --> 199,746 rows
# 10 --> 119,734 rows, 13:44-13:51, 7 minutes, t,b = 10
# 100 --> 21,606 rows, t,b = 10 1 minutes,
# 1,000 --> 3,087 rows, t = 1000, b = 10, t
# 10,000 --> 386 rows
# 100,000 --> 31 rows

# histogram: show which values in a vector occur how often
# my values are the counts
# The histogram shows: How frequent are very low/high/middle counts?
# y-axis: frequency = How often do values in this bin appear?
# x-axis: the magnitude of the counts, scale from 1 (a single time) to 4.47 million (sex or gender)

# HOW TO UNDERSTAND THE PLOT AND ITS VARIABLES
# the larger the binsize, the higher the counts --> more observations fall into this bin

plot = sns.histplot(data = tail_value_counts_df_filtered_for_counts,  # stat = 'count',
                    # binwidth = 10,
                    x = 'count')
plot.set_xscale('log')  # set x-axis to logscale
# plot.set_yscale('log')
y_axis_scale_manual = [1, 10, 100, 1000, 10000, 100000, 1000000]
# plot.set_yticks(x_axis_scale_manual)
# plot.set_yticklabels(x_axis_scale_manual)
plot.set_title(f'Count minimum: {threshold_for_minimum_count}')
plt.show()

# if it's a large plot, pickle it for later use
file = open('plot_counts_filter3,binsauto.pkl', 'wb')
pickle.dump(plot, file)
file.close()

# %% try out pylustrator

import pylustrator
pylustrator.start()




# %% Zoom in on specific areas
threshold_for_maximum_count = 10000
threshold_for_minimum_count = 10

tail_value_counts_df_filtered_for_counts = tail_value_counts_df.query(
    f'count>={threshold_for_minimum_count}')

tail_value_counts_df_filtered_for_counts = tail_value_counts_df_filtered_for_counts.query(
    f'count<={threshold_for_maximum_count}')



# %% TODO Transform counts into percentages

# needs to be done for the stacked percentage counts

# TODO DECISION: only keep counts >=10 to make the dataset more manageable
tail_value_counts_df = tail_value_counts_df.query('count>=10')

groupby_df = tail_value_counts_df.groupby(['relation_label', 'dataset_name'])

# use agg() function
groupby_df_agg = groupby_df.agg({
    'count': ['max', 'min', 'sum'],
    'tail_entity_Q_ID': 'size'
})
groupby_df_agg

# make columns that were sepcified for groupby distribute over all rows
groupby_df_agg = groupby_df_agg.reset_index()
# make normal columns out of the multiindex
groupby_df_agg.columns = ['_'.join(col).strip() for col in groupby_df_agg.columns.values]


# try this: https://stackoverflow.com/questions/40923165/python-pandas-equivalent-to-r-groupby-mutate

tail_value_counts_df['count_percentage'] = tail_value_counts_df.groupby(['relation_label', 'dataset_name']).apply(lambda col: col.count/col.count.sum())


# %% TODO Aggregate low percentages into OTHER category

threshold_for_OTHER_category = 10000
# there are 24 unique rows --> relations I want to look at

# It does not make sense to use a universal threshold for OTHER category!
# e.g. max count for 'native language' is 2125!

tail_value_counts_df.loc[tail_value_counts_df['count'] < threshold_for_OTHER_category]



# simple things first


# %% TODO Plot counts as stacked bar plots (percentage)

# one bar per dataset per relation
# color of the bar: Which dataset?
# length of the bar: Which percentage does this tail value have of the total counts
# annotate with absolute counts
# --> makes comparison between datasets better

# link: https://stackoverflow.com/questions/59038979/stacked-bar-chart-in-seaborn

ax = sns.histplot(data = tail_value_counts_df, y = 'dataset_name', hue = 'relation_label',
                  weights = 'percentage', multiple = 'stack', palette = 'tab20c', shrink = 0.8)
ax.set_ylabel('percentage')
