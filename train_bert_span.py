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
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp

from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    # ipython convenience
    mgc = get_ipython().magic
    mgc(u'%load_ext tensorboard')
    mgc(u'%load_ext autoreload')
    mgc(u'%autoreload 2')
    METHOD_NAME = 'dev/span_bert'
    LOG_DIR = "logs/" + METHOD_NAME
    os.chdir('/home/burtenshaw/now/spans_toxic')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name")
    parser.add_argument("--fold", default = 999, type=int)
    parser.add_argument('--hparams', action='store_true', default=False)
    parser.add_argument('--runs', default=1, type=int)
    args = parser.parse_args()

    METHOD_NAME = args.method_name
    # LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, args.fold)
    # print('logging at :', LOG_DIR)
    print('Method : ', METHOD_NAME)

# from results import EvalResults
from utils import *
from models import *
import params_config
from results import train_accuracy

tf.config.list_physical_devices(device_type='GPU')

#%%%
MAX_LEN = 128

if args.fold != 999:
    folds_list = [str(n) for n in range(args.fold)]
else:
    folds_list = ['all']

for fold in folds_list:

    data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
    output_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)
    save_path = os.path.join(output_dir, '%s.bin' % (METHOD_NAME))
    LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, fold)

    if os.path.isfile(save_path):
        continue

    print('logging at :', LOG_DIR)
    print('data_dir : ', data_dir)
    print('output_dir : ', output_dir)

    train = pd.read_pickle(os.path.join(data_dir, "train.bin"))
    val = pd.read_pickle(os.path.join(data_dir, "val.bin"))
    test = pd.read_pickle(os.path.join(data_dir, "test.bin"))

    print('train : ', train.shape)
    print('test : ', test.shape)
    print('val : ', val.shape)

    MAX_LEN = 128

    X_train = bert_prep(train.text.to_list(), max_len = MAX_LEN)
    X_val = bert_prep(val.text.to_list(), max_len = MAX_LEN)
    X_test = bert_prep(test.text.to_list(), max_len = MAX_LEN)

    y_train = np.vstack(pad_sequences(train.word_mask.values, 
                                    maxlen = MAX_LEN, 
                                    truncating = 'post', 
                                    padding = 'post'))

    y_val = np.vstack(pad_sequences(val.word_mask.values, 
                                    maxlen = MAX_LEN, 
                                    truncating = 'post', 
                                    padding = 'post'))

    y_test = np.vstack(pad_sequences(test.word_mask.values, 
                                    maxlen = MAX_LEN, 
                                    truncating = 'post', 
                                    padding = 'post'))

    #%%
    callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    METRICS = [
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy'),
    ]
    
    train_samples = {'X_train' : X_train, 
            'y_train' : y_train, 
            'X_val' : X_val, 
            'y_val' : y_val, 
            'X_test' : X_test, 
            'y_test' : y_test}

    if args.hparams:

        HPARAMS = [
                    hp.HParam('activation', hp.Discrete(['relu'])),
                    hp.HParam('batch_size', hp.Discrete([8,16])),
                    hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                    hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                    hp.HParam('n_layers', hp.Discrete([1])),
                    hp.HParam('model_scale',hp.Discrete([1,2,3])),
                    hp.HParam('epochs', hp.Discrete([2]))
                    ]

        with tf.summary.create_file_writer(LOG_DIR).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[hp.Metric(m.name, display_name=m.name) for m in METRICS],
            )

        for runs in range(args.runs):

            param_str = '%s_run_%s' % (METHOD_NAME, runs)
            run_dir = LOG_DIR + '/' + fold + param_str

            hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}

            print('_' * 80)
            for k, v in hparams.items():
                print('\t|%s = %s' % (k, v))
            print( ' = ' )
        
            callbacks.append(hp.KerasCallback(run_dir, hparams))

            with tf.summary.create_file_writer(run_dir).as_default():
                hp.hparams(hparams)  # record the values used in this trial
                
                results = bert_to_mask(data = train_samples,
                                    input_length = MAX_LEN,
                                    output_length = MAX_LEN,
                                    hparams = hparams, 
                                    callbacks = callbacks, 
                                    metrics = METRICS)

                print('_' * 80)
                for k, v in hparams.items():
                    print('\t|%s = %s' % (k, v))

                print( ' = ' )

                for metric, score in results.items():
                    print('\t|%s : %s' % (metric , score))
                    tf.summary.scalar(metric, score, step=1)
                
                print('_' * 80)

    else:

        hparams = params_config.BEST['BERT_SPAN']
        model = bert_to_mask(data = train_samples,
                input_length = MAX_LEN,
                output_length = MAX_LEN,
                hparams = hparams, 
                callbacks = callbacks, 
                metrics = METRICS,
                return_model = True
                )

        y_pred = model.predict(X_test)  
        pd.DataFrame(y_pred).to_pickle(save_path)
        # model.save(os.path.join(output_dir, '%s.model' % (METHOD_NAME)))

