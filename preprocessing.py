#%%
import os
import re
import json
import string
import numpy as np
import pandas as pd
from ast import literal_eval
from string import punctuation

os.chdir('/home/burtenshaw/now/spans_toxic')
from utils import *

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
df = pd.read_csv(data_dir + "toxic_train.csv")
df.spans = df.spans.apply(literal_eval)

max_len=1000
# %%
train = pd.DataFrame()
train['text'] = df.text
train['entities'] = df.spans
train['ent_check'] = df.spans.apply(lambda e : len(e)-(e[-1]-e[0]) if len(e) > 2 else np.nan)
#%%
train['cont_ranges'] = train.entities.apply(contiguous_ranges)
train['flat_cont_ranges'] = train.entities.apply(contiguous_ranges, flatten=True)
train['n_spans'] = train.cont_ranges.apply(len)
train['tokens'] = train.text.apply(lambda x : x.split(' '))
train['character_mask'] = train.entities.apply(make_character_mask)
train['word_mask'] = train.apply(make_word_mask, axis = 1)

# %% check word mask and character entity alignment 
train['predictions'] = train.apply(word_mask_to_character_entity, pad = False, axis = 1)
print(' parsing accuracy %s' % train.apply(lambda row : \
                            f1(row.predictions, row.entities), axis = 1).mean())

#%% add word level tuples of word and label
# check word mask and token alignment
print((train.tokens.apply(len) - train.word_mask.apply(len)).describe())

train['tuples'] = train[['word_mask', 'tokens']].apply(lambda row : list(zip(row.tokens, row.word_mask)), axis=1)

# %%
train.to_csv(data_dir + 'train.csv')
train.to_pickle(data_dir + 'train.bin')

# %%
