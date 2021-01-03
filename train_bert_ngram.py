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
    args = parser.parse_args()

    METHOD_NAME = args.method_name
    LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, args.fold)
    print('Method : ', METHOD_NAME)
    # print('logging at :', LOG_DIR)

# from results import EvalResults
from utils import *
from models import *
import params_config

tf.config.list_physical_devices(device_type='GPU')
#%%

if args.fold != 999:
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

    MAX_LEN = 200

    val['input_ids'], val['token_type_ids'], val['attn_mask'] = [x.tolist() for x in bert_prep(val.text.to_list(), max_len = MAX_LEN)]
    train['input_ids'], train['token_type_ids'], train['attn_mask'] = [x.tolist() for x in bert_prep(train.text.to_list(), max_len = MAX_LEN)]
    test['input_ids'], test['token_type_ids'], test['attn_mask'] = [x.tolist() for x in bert_prep(test.text.to_list(), max_len = MAX_LEN)]

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        AUC(name='auc')
    ]

    callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    if args.hparams:
        HPARAMS = [
                    hp.HParam('activation', hp.Discrete(['relu'])),
                    hp.HParam('batch_size', hp.Discrete([16])),
                    hp.HParam('lr', hp.Discrete([1e-3, 1e-5, 1e-7, 1e-9])),
                    hp.HParam('dropout',hp.Discrete([0.2])),
                    hp.HParam('n_layers', hp.Discrete([1,2,3])),
                    hp.HParam('model_scale',hp.Discrete([1,2,3])),
                    hp.HParam('pre', hp.Discrete([64])),
                    hp.HParam('post', hp.Discrete([64])),
                    hp.HParam('word', hp.Discrete([0])),
                    hp.HParam('epochs', hp.Discrete([2])),
                    hp.HParam('lstm', hp.Discrete([True]))
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

            for k, v in hparams.items():
                print('\t|%s = %s' % (k, v))
            
            pre, post, word = hparams['pre'], hparams['post'], hparams['word']

            X_train, y_train, _= make_BERT_context_data(train, pre = pre, post = post)
            X_val, y_val, _ = make_BERT_context_data(val, pre = pre, post = post)
            X_test, y_test, _ = make_BERT_context_data(test, pre = pre, post = post)

            train_samples = {'X_train' : X_train, 
                            'y_train' : y_train, 
                            'X_val' : X_val, 
                            'y_val' : y_val, 
                            'X_test' : X_test, 
                            'y_test' : y_test}


            callbacks.append(hp.KerasCallback(run_dir, hparams))

            with tf.summary.create_file_writer(run_dir).as_default():
                hp.hparams(hparams)  # record the values used in this trial
                
                results = ngram_dual_bert(data = train_samples,
                                            pre_length = pre + 1,
                                            post_length = post + 1,
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

    else:
        hparams = params_config.BEST['BERT_NGRAM']
        pre, post, word = hparams['pre'], hparams['post'], hparams['word']

        X_train, y_train, train_index = make_BERT_context_data(train, pre = pre, post = post)
        X_val, y_val, val_index = make_BERT_context_data(val, pre = pre, post = post)
        X_test, y_test, test_index = make_BERT_context_data(test, pre = pre, post = post)

        train_samples = {'X_train' : X_train, 
                        'y_train' : y_train, 
                        'X_val' : X_val, 
                        'y_val' : y_val, 
                        'X_test' : X_test, 
                        'y_test' : y_test}

        model = ngram_dual_bert(data = train_samples,
                                pre_length = pre + 1,
                                post_length = post + 1,
                                hparams = hparams, 
                                callbacks = callbacks, 
                                metrics = METRICS,
                                return_model = True
                )

        y_pred = model.predict(X_test)
        # model.save(os.path.join(output_dir, '%s.model' % (METHOD_NAME)))
        pd.DataFrame(y_pred, index = test_index).to_pickle(save_path)