#%%
import os
import re
import json
import string
import numpy as np
import pandas as pd
from ast import literal_eval
from string import punctuation

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
train['cont_ranges'] = train.apply(contiguous_ranges, axis=1)
train['flat_cont_ranges'] = train.apply(lambda x : contiguous_ranges(x, flatten=True), axis=1)
train['n_spans'] = train.span_ids.apply(len)

# %%
train['tokens'] = train.text.apply(lambda x : x.split(' '))
train['character_mask'] = train.entities.apply(make_character_mask)
train['word_mask'] = train.apply(make_word_mask, axis = 1)

# %%
train['predictions'] = train.apply(word_mask_to_character_entity, axis = 1)
train['f1'] = train.apply(lambda row : f1(row.predictions, row.entities), axis = 1)

# %%
train.to_csv(data_dir + 'train.csv')
train.to_pickle(data_dir + 'train.bin')
