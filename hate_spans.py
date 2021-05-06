#%%
import pandas as pd
import numpy as np
import os
import random
import datetime
import string
import re
import tempfile
import sys
import argparse
import json
from tqdm import *

from utils import f1

fold = 'all'
data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
# output_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)
# save_path = os.path.join(output_dir, '%s.json' % (METHOD_NAME))
# word_ids_path = os.path.join(output_dir, '%s.bin' % (METHOD_NAME + '_word_ids'))
# model_path = os.path.join(output_dir, '%s.bin' % (METHOD_NAME + '_model'))

train = pd.read_json(os.path.join(data_dir, "train.json"))
val = pd.read_json(os.path.join(data_dir, "val.json"))
test = pd.read_json(os.path.join(data_dir, "test.json"))
eval_ = pd.read_pickle(os.path.join(data_dir, "eval.bin"))
#%%

from mudes.app.mudes_app import MUDESApp

app = MUDESApp("en-large", use_cuda=True)
print(app.predict_toxic_spans("You motherfucking cunt", spans=True))

# %%
# test['pred_ents'] = test.text.apply(app.predict_toxic_spans, spans=True)
# test.to_json('data/hate_spans_test.json')

test = pd.read_json('data/hate_spans_test.json')
# %%
import string

SPECIAL_CHARACTERS = string.whitespace

def ents_to_spans(ents, text, special_characters=SPECIAL_CHARACTERS):
    """Applies minor edits to trim spans and remove singletons."""
    cleaned = []
    for begin, end in ents:
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned

test['pred_spans'] = test.apply(lambda row : ents_to_spans(row.pred_ents, row.text), axis = 1)

test['f1_score'] = test.apply(lambda row : f1(row.pred_spans, row.spans) , axis = 1)
print('span f1 : ', test.f1_score.mean())

# %%

eval_['pred_ents'] = eval_.text.apply(app.predict_toxic_spans, spans=True)
eval_.to_json('data/hate_spans_eval.json')
eval_['pred_spans'] = eval_.apply(lambda row : ents_to_spans(row.pred_ents, row.text), axis = 1)

out = eval_.pred_spans.to_list()
# %%    
from submit import to_submit

print('saving eval set')
to_submit(out, output_path='/home/burtenshaw/now/spans_toxic/predictions/submit/spans-pred_outer.txt')
# %%
