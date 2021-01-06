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
    METHOD_NAME = 'dev/ngram_bert'
    LOG_DIR = "logs/" + METHOD_NAME
    os.chdir('/home/burtenshaw/now/spans_toxic')
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name")
    parser.add_argument("--fold", default = 999, type=int)
    parser.add_argument('--hparams', action='store_true', default=False)
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--save_model', default=False, action='store_true')
    args = parser.parse_args()

    METHOD_NAME = args.method_name
    LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, args.fold)
    print('Method : ', METHOD_NAME)
    # print('logging at :', LOG_DIR)

# from results import EvalResults
from utils import *
from models import *
import config

tf.config.list_physical_devices(device_type='GPU')
#%%

if args.fold != 999 and args.method_name != 'ensemble':
    folds_list = [str(n) for n in range(args.fold)]
else:
    folds_list = ['all']

for fold in folds_list:

    data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
    output_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)
    save_path = os.path.join(output_dir, '%s.bin' % (METHOD_NAME))

    LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, fold)
    
    print('logging at :', LOG_DIR)
    print('data_dir : ', data_dir)
    print('output_dir : ', output_dir)

    train = pd.read_pickle(os.path.join(data_dir, "train.bin"))
    val = pd.read_pickle(os.path.join(data_dir, "val.bin"))
    test = pd.read_pickle(os.path.join(data_dir, "test.bin"))

    print('train : ', train.shape)
    print('val : ', val.shape)
    print('test : ', test.shape)
    
    method = config.METHODS[METHOD_NAME]
    method = method(train, val, test)

    callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    if args.hparams:
        HPARAMS = config.TUNING[METHOD_NAME]  

        with tf.summary.create_file_writer(LOG_DIR).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[hp.Metric(m.name, display_name=m.name) for m in method.metrics],
            )

        for runs in range(args.runs):
            
            param_str = '_%s_run_%s' % (METHOD_NAME, runs)
            run_dir = LOG_DIR + '/' + fold + param_str
            hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}

            method.callbacks = callbacks + [hp.KerasCallback(LOG_DIR, hparams)]
            method.hparams = hparams

            for k, v in hparams.items():
                print('\t|%s = %s' % (k, v))

            train_samples = method.get_data()

            with tf.summary.create_file_writer(run_dir).as_default():
                hp.hparams(hparams)  # record the values used in this trial
                
                results = method.run(train_samples)

                print('_' * 80)
                # print(ngram_str)
                for k, v in hparams.items():
                    print('\t|%s = %s' % (k, v))

                print( ' = ' )

                for metric, score in results.items():
                    print('\t|%s : %s' % (metric , score))
                    tf.summary.scalar(metric, score, step=1)
                
                print('_' * 80)

    else:
        hparams = config.BEST[METHOD_NAME]

        method = method(train, val, test)
        method.callbacks = callbacks + [hp.KerasCallback(LOG_DIR, hparams)]
        method.hparams = hparams
        
        train_samples = method.get_data()
        model = method.run(data = train_samples, return_model = True)
        y_pred = model.predict(train_samples['X_test'])
        
        if args.save_model:
            model.save(os.path.join(output_dir, '%s_model' % (METHOD_NAME)))
    
        pd.DataFrame(y_pred, index = method.test_index).to_pickle(save_path)
