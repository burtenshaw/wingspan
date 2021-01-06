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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import argparse

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from tensorboard.plugins.hparams import api as hp
import pickle

from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    # ipython convenience
    mgc = get_ipython().magic
    mgc(u'%load_ext tensorboard')
    mgc(u'%load_ext autoreload')
    mgc(u'%autoreload 2')
    METHOD_NAME = 'dev/categorical_ensemble'
    LOG_DIR = "logs/" + METHOD_NAME
    os.chdir('/home/burtenshaw/now/spans_toxic')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    class CheapArgs:
        hparams = False
        fold = 999
        method_name = METHOD_NAME
        runs = 1
        features = 'run'

    args = CheapArgs()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name')
    parser.add_argument('--fold', default = 999, type=int)
    parser.add_argument('--hparams', action='store_true', default=False)
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--features', default='run', type = str)
    args = parser.parse_args()
    METHOD_NAME = sys.method_name
    LOG_DIR = "logs/categorical/" + METHOD_NAME

os.chdir('/home/burtenshaw/now/spans_toxic')

from results import EvalResults
from utils import *
from models import *
import ensemble_features

tf.config.list_physical_devices(device_type='GPU')

# %%
''' 

predict word label based on : 

- model prediction for each word in sentence
- model prediction for current word
- lexical toxicity for full sentence
- lexical toxicity for word 

'''

MAX_LEN = 128

# features

folds_dir_list = ['0', '1', '2', '3', '4']

if args.features == 'run':
    X_train, y_train, X_test, y_test, text_spans = ensemble_features.get_data(folds_dir_list, test_text=True)
    
    data = [X_train, y_train, X_test, y_test, text_spans]

    with open('data/ensemble_features/data.bin', 'wb') as f:
        pickle.dump(data, f)

elif args.features == 'load':

    with open('data/ensemble_features/data.bin', 'rb') as f:
        data = pickle.load(f)

    X_train, y_train, X_test, y_test, text_spans = data
    test_text , test_spans = text_spans


def do_task_f1(text_list, true_spans, pred_masks):
    
    df = pd.DataFrame()
    df['text'] = text_list
    df['spans'] = true_spans
    df['pred_mask'] = pred_masks
    df['pred_spans'] = df.apply(spacy_word_mask_to_spans, field = 'pred_mask', axis = 1)
    df['f1_score'] = df.apply(lambda row : f1(row.pred_spans, row.spans), axis = 1)

    return df.f1_score.mean()


def ensemble_lstm(data, word_input_shape, sentence_input_shape, hparams, callbacks, metrics, task_f1 = False):  

    word_input = tf.keras.Input(shape=word_input_shape)
    word_lstm = layers.Bidirectional(layers.LSTM(128))(word_input)
    
    sentence_input = tf.keras.Input(shape=sentence_input_shape)
    sentence_dense = layers.Dense(128)(sentence_input)
    sentence_dense = layers.Dense(128)(sentence_dense)
    sentence_dense = layers.Bidirectional(layers.LSTM(128))(sentence_dense)

    layer = word_lstm * sentence_dense
    layer = layers.Dense(MAX_LEN)(layer)

    output = layers.Activation(K.activations.softmax)(layer)

    model = tf.keras.Model(
        inputs=[word_input, sentence_input],
        outputs=[output],
    )

    opt = Adam(lr = hparams['lr'])

    model.compile(optimizer = opt, 
                  loss = 'categorical_crossentropy', 
                  metrics = metrics)

    model.fit(  data['X_train'], 
                data['y_train'],
                batch_size=hparams['batch_size'],
                validation_split=0.2,
                epochs=hparams['epochs'],
                verbose = 1,
                callbacks= callbacks)

    
    scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
    
    if task_f1:
        y_pred = model.predict(data['X_test'])
        print(y_pred[0])
        y_pred = np.where(y_pred > 0.5, 1,0 )
        print(y_pred[0])
        print(test_spans[0])
        scores['task_f1'] = do_task_f1(test_text, test_spans, y_pred.tolist())

    
    return scores


HPARAMS = [
          hp.HParam('activation', hp.Discrete(['relu'])),
          hp.HParam('batch_size', hp.Discrete([32,64,128,256])),
          hp.HParam('lr', hp.Discrete([2e-2, 5e-5, 7e-7, 9e-9])),
          hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
          hp.HParam('n_layers', hp.Discrete([1,2])),
          hp.HParam('model_scale',hp.Discrete([1])),
          hp.HParam('epochs', hp.Discrete([30])),
          hp.HParam('word_lstm_nodes', hp.Discrete([24,48,64,128,256])),
          hp.HParam('sentence_lstm_nodes', hp.Discrete([24,48,64,128,256]))
          ]

METRICS = [
          tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
          ]

with tf.summary.create_file_writer(LOG_DIR).as_default():
    hp.hparams_config(
        hparams=HPARAMS,
        metrics=[hp.Metric(m.name, display_name=m.name) for m in METRICS],
    )

print('logging at :', LOG_DIR)

for runs in range(args.runs):

    hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}
    
    train_samples = {'X_train' : X_train, 
                     'y_train' : y_train, 
                     'X_test' : X_test, 
                     'y_test' : y_test}

    # ngram_str = '/pr_%s_w_%s_po_%s_' % (pre, word, post)
    # param_str = '_'.join(['%s_%s' % (k,v) for k,v in hparams.items()]).replace('.', '')
    param_str = '%s_run_%s' % (METHOD_NAME, runs)
    run_dir = LOG_DIR + '/' + param_str

    callbacks = [hp.KerasCallback(run_dir, hparams),
                 TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
                 EarlyStopping(patience=2)]

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        
        results = ensemble_lstm(data = train_samples,
                               word_input_shape = (128, 309), 
                               sentence_input_shape = (780,1),
                               hparams = hparams, 
                               callbacks = callbacks, 
                               metrics = METRICS,
                               task_f1 = True)

        print('_' * 80)
        # print(ngram_str)
        for k, v in hparams.items():
            print('\t|%s = %s' % (k, v))

        print( ' = ' )

        for metric, score in results.items():
            print('\t|%s : %s' % (metric , score))
            tf.summary.scalar(metric, score, step=1)
        
        print('_' * 80)

    now = datetime.datetime.now()
    runs += 1
    # %%
