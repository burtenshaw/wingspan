# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
import datetime

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC

import tensorflow_hub as hub
from tensorboard.plugins.hparams import api as hp

from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_curve
from sklearn.model_selection import RandomizedSearchCV, KFold

from tensorflow.keras.utils import to_categorical
os.chdir('/home/burtenshaw/now/spans_toxic')

# %load_ext tensorboard
# %load_ext autoreload
# %autoreload 2

from results import EvalResults
from utils import *
from models import *

import string
import re

import tempfile

from sklearn.preprocessing import StandardScaler

tf.config.list_physical_devices(device_type='GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

MAX_LEN = 128

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
data = pd.read_pickle(data_dir + "train.bin")

train_index, test_index = train_test_split(data.index.drop_duplicates(), test_size=0.2, random_state=2018)
train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=2018)

#%%
X_train_id, X_train_mask, X_train_attn = bert_prep(data.loc[train_index].text.to_list(), max_len = MAX_LEN)
X_val_id, X_val_mask, X_val_attn = bert_prep(data.loc[val_index].text.to_list(), max_len = MAX_LEN)
X_test_id, X_test_mask, X_test_attn = bert_prep(data.loc[test_index].text.to_list(), max_len = MAX_LEN)

y = to_categorical(data.len.values)
y_train = y[train_index]
y_val = y[val_index]
y_test =y[test_index]
#%%
METHOD_NAME = 'categorical_len'
LOG_DIR = "logs/" + METHOD_NAME

HPARAMS = [
          hp.HParam('activation', hp.Discrete(['relu', 'tanh'])),
          hp.HParam('batch_size', hp.Discrete([8,16,32])),
          hp.HParam('lr', hp.Discrete([0.001, 0.01, 0.1])),
          hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
          hp.HParam('n_layers', hp.Discrete([1,2,3,4])),
          hp.HParam('model_scale',hp.Discrete([1,2])),
          hp.HParam('epochs', hp.Discrete([2]))
          ]

METRICS = [
          TruePositives(name='tp'),
          FalsePositives(name='fp'),
          TrueNegatives(name='tn'),
          FalseNegatives(name='fn'), 
          BinaryAccuracy(name='binary_accuracy'),
          Precision(name='precision'),
          Recall(name='recall'),
          AUC(name='auc')
]

with tf.summary.create_file_writer(LOG_DIR).as_default():
    hp.hparams_config(
        hparams=HPARAMS,
        metrics=[hp.Metric(m.name, display_name=m.name) for m in METRICS],
    )

print('logging at :', LOG_DIR)
now = datetime.datetime.now()
tomorrow = now + datetime.timedelta(days=1)


#%%
while now < tomorrow:

    hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}
    
    train_samples = {'X_train' : [X_train_id, X_train_mask, X_train_attn], 
                     'y_train' : y_train, 
                     'X_val' : [X_val_id, X_val_mask, X_val_attn], 
                     'y_val' : y_val, 
                     'X_test' : [X_test_id, X_test_mask, X_test_attn], 
                     'y_test' : y_test}

    # ngram_str = '/pr_%s_w_%s_po_%s_' % (pre, word, post)
    param_str = '_'.join(['%s_%s' % (k,v) for k,v in hparams.items()]).replace('.', '')

    run_dir = LOG_DIR + '/' + param_str

    callbacks = [hp.KerasCallback(run_dir, hparams),
                TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        
        results = bert_to_mask(data = train_samples,
                               input_length = MAX_LEN,
                               output_length = y_val[0].shape[0],
                               hparams = hparams, 
                               callbacks = callbacks, 
                               metrics = METRICS,
                               loss = 'categorical_crossentropy',)

        print('_' * 80)
        # print(ngram_str)
        for k, v in hparams.items():
            print('\t|%s = %s' % (k, v))

        print( ' = ' )

        for metric, score in results.items():
            print('\t|%s : %s' % (metric , score))
            tf.summary.scalar(metric, score, step=1)
        
        print('_' * 80)
    
    print('\n accuracy : %s' % results)

    now = datetime.datetime.now()
    # %%
