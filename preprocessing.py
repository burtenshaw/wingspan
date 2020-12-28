#%%
import os
import re
import json
import string
import numpy as np
import pandas as pd
from ast import literal_eval
from string import punctuation
from sklearn.model_selection import train_test_split

os.chdir('/home/burtenshaw/now/spans_toxic')
from utils import *

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
df = pd.read_csv(data_dir + "TOXIC_SPANS_ORIGINAL.csv")
df.spans = df.spans.apply(literal_eval)

all_data = pd.DataFrame()
all_data['text'] = df.text
all_data['spans'] = df.spans
all_data['ent_check'] = df.spans.apply(lambda e : len(e)-(e[-1]-e[0]) if len(e) > 2 else np.nan)
all_data['cont_ranges'] = all_data.spans.apply(contiguous_ranges)
all_data['flat_cont_ranges'] = all_data.spans.apply(contiguous_ranges, flatten=True)
all_data['tokens'] = all_data.text.apply(lambda x : x.split(' '))
all_data['character_mask'] = all_data.spans.apply(make_character_mask)
all_data['word_mask'] = all_data.apply(make_word_mask, axis = 1)

# categorical targets
all_data['n_spans'] = all_data.cont_ranges.apply(len)
all_data['start'] = all_data.spans.apply(lambda x : x[0] if len(x) > 0 else -1)
all_data['end'] = all_data.spans.apply(lambda x : x[-1] if len(x) > 0 else -1)
all_data['len'] = all_data.spans.apply(lambda x : x[-1] - x[0] if len(x) > 0 else 0)

# %% check word mask and character entity alignment 
all_data['parsing_predictions'] = all_data.apply(word_mask_to_character_entity, pad = False, axis = 1)
print(' parsing accuracy %s' % all_data.apply(lambda row : \
                            f1(row.parsing_predictions, row.spans), axis = 1).mean())

#%% add word level tuples of word and label
# check word mask and token alignment
print((all_data.tokens.apply(len) - all_data.word_mask.apply(len)).describe())

all_data['tuples'] = all_data[['word_mask', 'tokens']].apply(lambda row : list(enumerate(zip(row.tokens, row.word_mask))), axis=1)

# %%
all_data.to_csv(data_dir + 'all_data.csv')
all_data.to_pickle(data_dir + 'all_data.bin')
# %%
train_index, test_index = train_test_split(all_data.index, test_size=0.2, random_state=2018)
train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=2018)

train = all_data.loc[train_index]
val = all_data.loc[val_index]
test = all_data.loc[test_index]

train.to_csv(data_dir + 'train.csv')
train.to_pickle(data_dir + 'train.bin')

val.to_csv(data_dir + 'val.csv')
val.to_pickle(data_dir + 'val.bin')

test.to_csv(data_dir + 'test.csv')
test.to_pickle(data_dir + 'test.bin')
# %%
