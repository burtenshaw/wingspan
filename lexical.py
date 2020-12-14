# %%
import pandas as pd
import numpy as np
import spacy
import os
from sklearn import metrics
os.chdir('/home/burtenshaw/now/spans_toxic')

%load_ext autoreload
%autoreload 2

from results import EvalResults
from utils import *
from models import *

data_dir = '/home/burtenshaw/now/spans_toxic/data/'

data = pd.read_pickle(data_dir + "train.bin")
data = data.loc[data.word_mask.apply(np.array).apply(sum) > 1]
data = make_word_level_df(data)
#%%

nlp = spacy.load("en")

# %%
TOXIC_WORDS = list(set(data.loc[data.word.map(lambda x : nlp.vocab[x].is_stop == False) & (data.label == 1)].word.to_list()))
data['lexical'] = data.word.apply(lambda x : 1 if x in TOXIC_WORDS else 0 )
#%%
data.to_pickle('data/lexical_pred.bin')
# %%
r = pd.DataFrame(metrics.classification_report(data.label, data.lexical, output_dict=True))
# %%

data = pd.read_pickle(data_dir + "train.bin")

data['lexical_predict'] = data.text.apply(lambda sentence : \
    [1 if word in TOXIC_WORDS else 0 for word in sentence.split(' ')])
# %%
max_tags = 239
preds = np.vstack(data.lexical_predict.apply(np.array).apply(pad_mask, max_len = max_tags).to_list())
true = np.vstack(data.word_mask.apply(np.array).apply(pad_mask, max_len = max_tags).to_list())
lex_e = EvalResults(preds, true, roc = False, token_type=None)
# %%
lex_e.label_scores()
# %%
