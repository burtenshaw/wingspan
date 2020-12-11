# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_curve

os.chdir('/home/burtenshaw/now/spans_toxic')
from results import EvalResults
from utils import *
from models import *

max_len = 200

data_dir = '/home/burtenshaw/now/spans_toxic/data/'

data = pd.read_pickle(data_dir + "train.bin")

# %%
X_train, X_test, y_train, y_test, X_mask, X_ids, max_tags = prep_data(data, 'word_mask')
#%%

bert_model = build_bert(output_dim=max_tags)
bert_model.fit([X_ids , X_mask] , y_train,
            validation_split=0.1,
            batch_size = 12, 
            epochs=3,
)

X_mask,X_ids = bert_prep(X_test)
results= bert_model.predict([X_ids,X_mask])

#%%
bert_e = EvalResults(results, y_test, roc = True)

#%%
embedding_matrix, num_tokens, vectorizer  = muse_embedding(X_train)
x_train, x_val, y_train, y_val  = muse_prep(X_train, X_test, y_train, y_test, vectorizer)
muse_model=build_muse_lstm(embedding_matrix, num_tokens, output_dim=max_tags)
muse_model.fit(x_train, y_train, batch_size=128, epochs=8, validation_split=0.1)

#%%
results = muse_model.predict(x_val)

#%%
muse_e = EvalResults(results, y_test, roc = True)
