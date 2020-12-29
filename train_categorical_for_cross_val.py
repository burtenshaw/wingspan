# %%
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

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import tensorflow_hub as hub
from tensorboard.plugins.hparams import api as hp

from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification

from tensorflow.keras.utils import to_categorical

from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    # ipython convenience
    mgc = get_ipython().magic
    mgc(u'%load_ext tensorboard')
    mgc(u'%load_ext autoreload')
    mgc(u'%autoreload 2')
    METHOD_NAME = 'dev/categorical_start'
    LOG_DIR = "logs/" + METHOD_NAME
    os.chdir('/home/burtenshaw/now/spans_toxic')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
    LOG_DIR = "logs/categorical/" + METHOD_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("method_name")
    parser.add_argument("fold", default='', type=str)
    parser.add_argument("hparam", default = False, type=bool)
    args = parser.parse_args()
    METHOD_NAME = args.method_name

from results import EvalResults
from utils import *
from models import *

tf.config.list_physical_devices(device_type='GPU')

#%%%

data_dir = '/home/burtenshaw/now/spans_toxic/data/%s' % args.fold
output_dir = '/home/burtenshaw/now/spans_toxic/data/predictions/%s' % args.fold

train = pd.read_pickle(data_dir + "train.bin")
val = pd.read_pickle(data_dir + "val.bin")
test = pd.read_pickle(data_dir + "test.bin")

MAX_LEN = 128
OUTPUT_LENGTH = max(train.start.max(), val.start.max(), test.start.max()) + 1

X_train = bert_prep(train.text.to_list(), max_len = MAX_LEN)
X_val = bert_prep(val.text.to_list(), max_len = MAX_LEN)
X_test = bert_prep(test.text.to_list(), max_len = MAX_LEN)

y_train = to_categorical(train.start.values, num_classes = OUTPUT_LENGTH)
y_val = to_categorical(val.start.values, num_classes = OUTPUT_LENGTH)
y_test = to_categorical(test.start.values, num_classes = OUTPUT_LENGTH)

#%%
print('logging at :', LOG_DIR)

METRICS = [
          tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
          ]

if args.hparam:
    HPARAMS = [
            hp.HParam('activation', hp.Discrete(['relu'])),
            hp.HParam('batch_size', hp.Discrete([8,16,32])),
            hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
            hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
            hp.HParam('n_layers', hp.Discrete([1,2])),
            hp.HParam('model_scale',hp.Discrete([1,2])),
            hp.HParam('epochs', hp.Discrete([10])),
            ]


    with tf.summary.create_file_writer(LOG_DIR).as_default():
        hp.hparams_config(
            hparams=HPARAMS,
            metrics=[hp.Metric(m.name, display_name=m.name) for m in METRICS],
        )

    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=0.5)

else:
    best_params = pd.read_csv('best_params.csv')
    hparams = dict(best_params.loc[METHOD_NAME])
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(minutes=5)

runs = 0
#%%
while now < tomorrow:
    param_str = '%s_run_%s' % (METHOD_NAME, runs)
    run_dir = LOG_DIR + '/' + param_str
    callbacks = [
                TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
                EarlyStopping(patience=2)]

    train_samples = {'X_train' : X_train, 
                     'y_train' : y_train, 
                     'X_val' : X_val, 
                     'y_val' : y_val, 
                     'X_test' : X_test, 
                     'y_test' : y_test}

    if args.hparams:
        hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}
        callbacks.append([hp.KerasCallback(run_dir, hparams)])


    # ngram_str = '/pr_%s_w_%s_po_%s_' % (pre, word, post)
    # param_str = '_'.join(['%s_%s' % (k,v) for k,v in hparams.items()]).replace('.', '')


        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            
            results = categorical_bert(data = train_samples,
                                input_length = MAX_LEN,
                                output_length = y_val[0].shape[0],
                                hparams = hparams, 
                                callbacks = callbacks, 
                                metrics = METRICS,
                                loss = 'categorical_crossentropy')

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
        
        else:
            model = categorical_bert(data = train_samples,
                    input_length = MAX_LEN,
                    output_length = y_val[0].shape[0],
                    hparams = hparams, 
                    callbacks = callbacks, 
                    metrics = METRICS,
                    loss = 'categorical_crossentropy',
                    return_model = True
                    )

            y_pred = model.predict(X_test)
            test[METHOD_NAME] = y_pred
            test[[METHOD_NAME]].to_pickle('%s/%s.bin' % (output_dir, METHOD_NAME))

    now = datetime.datetime.now()
    runs += 1
    # %%