# %%
import pandas as pd
import numpy as np
import spacy
import os
from sklearn import metrics
os.chdir('/home/burtenshaw/now/spans_toxic')
from ast import literal_eval
import spacy
from spacy.lang.en import English
from utils import spans_to_ents

%load_ext autoreload
%autoreload 2

from results import EvalResults
from utils import *
from models import *

data_dir = '/home/burtenshaw/now/spans_toxic/data/all/'

test = pd.read_pickle(data_dir + "eval.bin")
labels = pd.read_csv('data/tsd_test.csv')
test['spans'] = labels.spans.apply(literal_eval)
test['cont_ranges'] = test.spans.apply(contiguous_ranges)

#%%

df = pd.DataFrame()

for f in os.listdir('data/all'):
    if '.json' in f:
        p = os.path.join('data/all', f)
        _df = pd.read_json(p)
        df = pd.concat([df,_df])

#%%

# from transformers import ElectraTokenizerFast
# tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
# idx_word = {v:k for k,v in tokenizer.vocab.items()}

# def tokenate(text):
#     encoding = tokenizer.encode_plus(text, 
#                                     max_length = 200, 
#                                     truncation=True, 
#                                     padding='max_length')
#     back = [idx_word[i] for i in encoding['input_ids']]

#     return [w for w in back if w not in ['[CLS]', '[SEP]', '[PAD]']]

def get_toxic_words(row):
    tw = ''
    for start, end in row.cont_ranges:
        tw += row.text[start:end+1] + ' '
    return tw

df['toxic_words'] = df.apply(get_toxic_words, axis = 1)


nlp_base = English()
tokenizer = nlp_base.tokenizer

df['toxic_words'] = df.toxic_words.apply(tokenizer)
# df['toxic_words'] = df.toxic_words.apply(list)
# lambda s : [w for w in nlp(s) if nlp.vocab[w] == False]
#%%
nlp = spacy.load('en_core_web_sm')
STOP_WORDS = nlp.Defaults.stop_words
STOP_WORDS = [str(w) for w in STOP_WORDS]
df.toxic_words = df.toxic_words.apply(lambda x : [str(w) for w in x if str(w).lower() not in STOP_WORDS] )
TOXIC_WORDS = list(set(df.iloc[:5800].loc[df.toxic_words.apply(len) == 1].toxic_words.explode().to_list()))
TOXIC_WORDS = [str(w).lower() for w in TOXIC_WORDS]
TOXIC_WORDS = [w for w in TOXIC_WORDS if w not in STOP_WORDS]
#%%

# test['tokens'] = test.text.apply(tokenate)

import string
SPECIAL_CHARACTERS = string.whitespace

def lexical_predict(row):
    c = []
    p = 0
    for t in row.tokens:
        if t.lower() in TOXIC_WORDS:
            c.extend(range(p,p+len(t)))
        p = p + len(t) + 1

    c_c = []

    for _c in c:
        try:
            l = row.text[_c]
            if l not in SPECIAL_CHARACTERS and l.isalpha():
                c_c.append(_c)
        except IndexError:
            pass

    return c_c

test['pred'] = test.apply(lexical_predict, axis=1)

# test['clean_pred'] = test.apply(lambda r : fix_spans(r.pred, r.text), axis = 1)
# %%

test['f1'] = test.apply(lambda r : f1(r.pred, r.spans), axis=1)
test.f1.mean()

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
