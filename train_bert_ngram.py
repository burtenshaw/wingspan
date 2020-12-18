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
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
data = pd.read_pickle(data_dir + "train.bin")# data = data.loc[data.word_mask.apply(np.array).apply(sum) > 2]

METHOD_NAME = 'ngram_bert'
LOG_DIR = "logs/" + METHOD_NAME

NGRAM_SMALLEST = 50
NGRAM_LARGEST = 100

HPARAMS = [
          hp.HParam('activation', hp.Discrete(['relu'])),
          hp.HParam('batch_size', hp.Discrete([8,16])),
          hp.HParam('lr', hp.Discrete([0.001, 0.01, 0.1])),
          hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
          hp.HParam('n_layers', hp.Discrete([1,2,3,4])),
          hp.HParam('model_scale',hp.Discrete([1,2,3,4])),
          hp.HParam('pre', hp.Discrete(range(NGRAM_SMALLEST, NGRAM_LARGEST))),
          hp.HParam('post', hp.Discrete(range(NGRAM_SMALLEST, NGRAM_LARGEST))),
          hp.HParam('word', hp.Discrete([0])),
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

while now < tomorrow:

    hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}
    
    pre, post, word = hparams['pre'], hparams['post'], hparams['word']

    X_y = data.apply(make_context_labelling, axis = 1, pre = pre, post = post, word = word)\
        .explode().dropna().apply(pd.Series)

    X_y.columns = ['pre', 'word','post','label']
    
    X_train, y_train, X_val, y_val, X_test, y_test = make_BERT_context_data(X_y, pre = pre, post = post)

    train_samples = {'X_train' : X_train[0] + X_train[1], 
                     'y_train' : y_train, 
                     'X_val' : X_val[0] + X_val[1], 
                     'y_val' : y_val, 
                     'X_test' : X_test[0] + X_test[1], 
                     'y_test' : y_test}

    # ngram_str = '/pr_%s_w_%s_po_%s_' % (pre, word, post)
    param_str = '_'.join(['%s_%s' % (k,v) for k,v in hparams.items()]).replace('.', '')

    run_dir = LOG_DIR + '/' + param_str

    callbacks = [hp.KerasCallback(run_dir, hparams),
                TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        
        results = ngram_dual_bert(data = train_samples,
                                    pre_length = pre,
                                    post_length = post,
                                    hparams = hparams, 
                                    callbacks = callbacks, 
                                    metrics = METRICS)

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
# rdf = data.loc[test_index]
# rdf.columns = pd.MultiIndex.from_product([['base'],rdf.columns])
# rdf = X_y.loc[test_index]
# rdf.columns = pd.MultiIndex.from_product([['base'],rdf.columns])
# runs = []
# rdf['lstm_ngram_%s' % n, 'pred'] = model.predict(X_test)
# used_params['lstm_ngram_%s' % n] = p
# r = EvalResults([ (m, 0.5) for m in used_params.keys()], rdf)
# r.rdf
# %%