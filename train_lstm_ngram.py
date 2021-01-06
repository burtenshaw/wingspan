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
    METHOD_NAME = 'dev/ngram_glove_lstm'
    LOG_DIR = "logs/" + METHOD_NAME
    os.chdir('/home/burtenshaw/now/spans_toxic')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    class CheapArgs:
        hparams = False
        fold = 'all'
        method_name = METHOD_NAME

    args = CheapArgs()
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
    print('logging at :', LOG_DIR)

# from results import EvalResults
from utils import *
from models import *
import params_config

tf.config.list_physical_devices(device_type='GPU')

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
        print('fold done!', save_path)
        continue

    print('logging at :', LOG_DIR)
    print('data_dir : ', data_dir)
    print('output_dir : ', output_dir)

    train = pd.read_pickle(os.path.join(data_dir, "train.bin"))
    test = pd.read_pickle(os.path.join(data_dir, "test.bin"))
    val = pd.read_pickle(os.path.join(data_dir, "val.bin"))

    print('train : ', train.shape)
    print('test : ', test.shape)
    print('val : ', val.shape)
    #%%

    index_word = dict(enumerate(set(train.tokens.explode().to_list() + \
                                val.tokens.explode().to_list() + \
                                test.tokens.explode().to_list())))

    word_index =  dict(map(reversed, index_word.items()))

    train['sequences'] = train.tokens.apply(lambda sentence : [word_index[w] for w in sentence]).to_list()
    test['sequences'] = test.tokens.apply(lambda sentence : [word_index[w] for w in sentence]).to_list()
    val['sequences'] = val.tokens.apply(lambda sentence : [word_index[w] for w in sentence]).to_list()
    embedding_matrix = get_embedding_weights(word_index)

    #%%

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]

    callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    if args.hparams:
        HPARAMS = [
                    hp.HParam('activation', hp.Discrete(['relu'])),
                    hp.HParam('batch_size', hp.Discrete([32,64,128,256])),
                    hp.HParam('lr', hp.Discrete([0.001, 0.005, 0.01,])),
                    hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                    hp.HParam('n_layers', hp.Discrete([1,2,3])),
                    hp.HParam('model_scale',hp.Discrete([1,2])),
                    hp.HParam('pre', hp.Discrete(range(10, 60))),
                    hp.HParam('post', hp.Discrete(range(10, 60))),
                    hp.HParam('word', hp.Discrete([1])),
                    hp.HParam('epochs', hp.Discrete([6]))
                    ]

        with tf.summary.create_file_writer(LOG_DIR).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[hp.Metric(m.name, display_name=m.name) for m in METRICS],
            )

        for runs in range(args.runs):

            param_str = '%s_run_%s' % (METHOD_NAME, runs)
            run_dir = LOG_DIR + '/' + METHOD_NAME + fold + param_str

            if args.hparams:


                hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}
                callbacks.append(hp.KerasCallback(run_dir, hparams))
                
                X_train, y_train, _ = make_context_data(train, 
                                                    pre = hparams['pre'], 
                                                    post = hparams['post'], 
                                                    word = hparams['word'])
                
                X_val, y_val, _ = make_context_data(val, 
                                                pre = hparams['pre'], 
                                                post = hparams['post'], 
                                                word = hparams['word'])
                
                X_test, y_test, _ = make_context_data(test, 
                                                pre = hparams['pre'], 
                                                post = hparams['post'], 
                                                word = hparams['word'])

                train_samples = {'X_train' : X_train, 
                                'y_train' : y_train, 
                                'X_val' : X_val, 
                                'y_val' : y_val, 
                                'X_test' : X_test, 
                                'y_test' : y_test, 
                                'embedding_matrix' : embedding_matrix}


                # ngram_str = '/pr_%s_w_%s_po_%s_' % (pre, word, post)


                with tf.summary.create_file_writer(run_dir).as_default():
                    hp.hparams(hparams)  # record the values used in this trial
                    
                    results = ngram_glove_lstm(data = train_samples,
                                                pre_length = hparams['pre'],
                                                word_length=1,
                                                post_length = hparams['post'],
                                                hparams = hparams, 
                                                callbacks = callbacks, 
                                                metrics = METRICS,
                                                embedding_matrix=embedding_matrix)
                    print('_' * 80)
                    # print(ngram_str)
                    for k, v in hparams.items():
                        print('\t|%s = %s' % (k, v))

                    print(' = ')

                    for metric, score in results.items():
                        print('\t|%s : %s' % (metric , score))
                        tf.summary.scalar(metric, score, step=1)
                    
                    print('_' * 80)
    else:
        hparams = params_config.BEST['LSTM_NGRAM']

        X_train, y_train, train_index  = make_context_data(train, 
                                            pre = hparams['pre'], 
                                            post = hparams['post'], 
                                            word = hparams['word'])
        
        X_val, y_val, val_index  = make_context_data(val, 
                                        pre = hparams['pre'], 
                                        post = hparams['post'], 
                                        word = hparams['word'])
        
        X_test, y_test, test_index  = make_context_data(test, 
                                        pre = hparams['pre'], 
                                        post = hparams['post'], 
                                        word = hparams['word'])

        train_samples = {'X_train' : X_train, 
                        'y_train' : y_train, 
                        'X_val' : X_val, 
                        'y_val' : y_val, 
                        'X_test' : X_test, 
                        'y_test' : y_test, 
                        'embedding_matrix' : embedding_matrix}

        model = ngram_glove_lstm(data = train_samples,
                                    pre_length = hparams['pre'],
                                    word_length=1,
                                    post_length = hparams['post'],
                                    hparams = hparams, 
                                    callbacks = callbacks, 
                                    metrics = METRICS,
                                    embedding_matrix=embedding_matrix,
                                    return_model = True
                                    )

        y_pred = model.predict(X_test)
        pd.DataFrame(y_pred, index = test_index).to_pickle(save_path)
        # model.save(os.path.join(output_dir, '%s.model' % (METHOD_NAME)))

