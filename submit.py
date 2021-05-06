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
from ast import literal_eval
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
os.chdir('/home/burtenshaw/now/spans_toxic')
# import spacy
# nlp = spacy.load("en_core_web_sm")
# analyser = SentimentIntensityAnalyzer()
from utils import spacy_word_mask_to_spans, f1

# %%
# METHOD_NAME = 'token_bert'

# pred_path = '/home/burtenshaw/now/spans_toxic/predictions/all'

# with open(os.path.join(pred_path, '%s.bin' % (METHOD_NAME)), 'rb') as f:
#     preds = np.load(f, allow_pickle=True)

# with open(os.path.join(pred_path, '%s_%s.bin' % (METHOD_NAME, 'word_ids')), 'rb') as f:
#     word_ids = np.load(f, allow_pickle=True)

# data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', 'all')
# test = pd.read_pickle(os.path.join(data_dir, "test.bin"))

def align_single_predictions(y_pred, word_ids):

    w_preds = []

    for p, w in zip(y_pred,word_ids):
        if w == None:
            continue
        if p == 1:
            w_preds.append(w)

    w_preds = list(set(w_preds))

    a_preds = np.zeros(200)

    for w in w_preds:
        a_preds[w] = 1

    return w_preds, a_preds

def align_predictions(y_pred, word_ids, attn_mask):

    w_id_predictions = []
    aligned_predictions = []

    for pr, wids, attn in zip(y_pred,word_ids, attn_mask):

        w_preds = []

        for p, w, a in zip(pr,wids, attn):
            if a != 0:
                if w == None:
                    continue
                if p == 1:
                    w_preds.append(w)
            else:
                break

        w_preds = list(set(w_preds))
        w_id_predictions.append(w_preds)

        a_preds = np.zeros(200)

        for w in w_preds:
            a_preds[w] = 1

        aligned_predictions.append(a_preds)

    return w_id_predictions, aligned_predictions

# y_pred = list(np.argmax(np.vstack(y_pred)[:,:,:2], 2))
# w_id_predictions, aligned_predictions = align_predictions(y_pred, method.test.word_ids)


# true = [t[:128] for t in test.word_mask.to_list()]
# true = [np.where(t == 1)[0] for t in true]
# pred = w_id_predictions

# all_fs = [f1(p,t) for t,p in zip(true,pred)]
# print('Word level f1 : ', np.mean(all_fs))
# test['y_preds'] = aligned_predictions
# test['pred_spans'] = test.apply(spacy_word_mask_to_spans, field = 'y_preds', axis = 1)
# #%%
# test['f1_score'] = test.apply(lambda row : f1(row.pred_spans, row.spans) , axis = 1)
# print('span f1 : ', test.f1_score.mean())

        # %%

def to_submit(span_list, output_path = 'spans-pred.txt'):
    output = ''
    for n, line in enumerate(span_list):
        output += '%s\t%s\n' % (n,line)

    with open(output_path, 'w') as f:
        f.write(output)

# to_submit(data.predictions.to_list())