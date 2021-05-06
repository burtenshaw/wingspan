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
from datetime import datetime

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
    parser.add_argument("--fold", default = 99, type=int)
    parser.add_argument("--start", default = 0, type=int)
    parser.add_argument('--hparams', action='store_true', default=False)
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    args = parser.parse_args()

    METHOD_NAME = args.method_name.upper()
    LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, args.fold)
    print('Method : ', METHOD_NAME)
    # print('logging at :', LOG_DIR)

from utils import *
import config
from models import to_ensemble
from utils import spacy_word_mask_to_spans, f1
from submit import align_predictions,  to_submit

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#%%

if args.fold != 99 and args.method_name != 'ensemble':
    folds_list = [str(n) for n in range(args.start, args.fold)]
else:
    folds_list = ['all']

for fold in folds_list:

    data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
    output_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)
    model_dir = os.path.join('/home/burtenshaw/now/spans_toxic/models', fold)
    save_path = os.path.join(output_dir, '%s.json' % (METHOD_NAME))

    LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, fold)

    now = datetime.utcnow()
    easy_path = os.path.join('easy_logs', "%s_%s_%s.json" % (METHOD_NAME, fold, now))
    easy = pd.DataFrame()
    
    # print('logging at :', LOG_DIR)
    print('data_dir : ', data_dir)
    print('output_dir : ', output_dir)

    train = pd.read_json(os.path.join(data_dir, "train.json"))
    val = pd.read_json(os.path.join(data_dir, "val.json"))
    test = pd.read_json(os.path.join(data_dir, "test.json"))

    print('train : ', train.shape)
    print('val : ', val.shape)
    print('test : ', test.shape)
    

    # callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    for runs in range(args.runs):

        method = config.METHODS[METHOD_NAME]
        method = method(train, val, test, method_name = METHOD_NAME)

        if args.hparams:
            HPARAMS = config.TUNING[METHOD_NAME]  

            # with tf.summary.create_file_writer(LOG_DIR).as_default():
            #     hp.hparams_config(
            #         hparams=HPARAMS,
            #         metrics=[hp.Metric(m.name, display_name=m.name) for m in method.metrics] + [hp.Metric('task_f1', display_name='task_f1')],
            #     )

                
            param_str = '_%s_run_%s' % (METHOD_NAME, runs)
            run_dir = LOG_DIR + '/' + fold + param_str
            hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}

            # method.callbacks = callbacks + [hp.KerasCallback(LOG_DIR, hparams)]
            method.callbacks = []
            
            method.hparams = hparams

            for k, v in hparams.items():
                print('\t|%s = %s' % (k, v))
                

            train_samples = method.get_data()

            # with tf.summary.create_file_writer(run_dir).as_default():
            #     hp.hparams(hparams)  # record the values used in this trial
                
            results = method.run(train_samples)

            print('_' * 80)
            # print(ngram_str)
            for k, v in hparams.items():
                print('\t|%s = %s' % (k, v))
                easy.at[runs, k] = v

            print( ' = ' )

            for metric, score in results.items():
                print('\t|%s : %s' % (metric , score))
                # tf.summary.scalar(metric, score, step=1)
                easy.at[runs, metric] = score
            
            print('_' * 80)

            easy.to_json(easy_path)

        else:
            hparams = config.BEST[(METHOD_NAME, fold)]
            # method.callbacks = callbacks + [hp.KerasCallback(LOG_DIR, hparams)]
            method.callbacks = []
            method.hparams = hparams
            
            train_samples = method.get_data()

            print('fold : ', fold)
            print('output_dir : ', output_dir)
            
            for k, v in hparams.items():
                print('\t|BEST : %s = %s' % (k, v))

            model = method.run(data = train_samples, return_model = True)
            
            # predict and save for ensemble

            y_pred = model.predict(method.X_test)
            to_ensemble(y_pred, method, output_dir)

            # log params and scores
            hparams.update(method.scores)
            for k, v in hparams.items():
                easy.at[0, k] = v

            easy.to_json(easy_path)

            # save the model
            if args.save_model:
                model.save(os.path.join(model_dir, '%s_%s' % (METHOD_NAME, method.task_score)))
            
        if args.eval:

            y_pred = method.y_pred
            model = method.model
            
            pred = list(np.argmax(y_pred[:,:,:], 2))
            w_id_predictions, aligned_predictions = align_predictions(pred, method.test.word_ids, method.test.attn_mask)
            true = method.test.word_mask.apply(lambda x : np.where(np.array(x)== 1)[0]).to_list()
            pred = w_id_predictions

            all_fs = [f1(p,t) for t,p in zip(true,pred)]
            print('Word level f1 : ', np.mean(all_fs))
            test['y_preds'] = aligned_predictions
            test['pred_spans'] = test.apply(spacy_word_mask_to_spans, field = 'y_preds', axis = 1)

            test['f1_score'] = test.apply(lambda row : f1(row.pred_spans, row.spans) , axis = 1)
            print('span f1 : ', test.f1_score.mean())

            eval_ = pd.read_pickle(os.path.join(data_dir, "eval.bin"))
            X_eval = method.evaluation_data(eval_)
            y_eval = model.predict(X_eval)
            y_eval = list(np.argmax(y_eval[:,:,:], 2))
            _, eval_['y_preds'] = align_predictions(y_eval, eval_.word_ids, eval_.attn_mask)
            eval_['pred_spans'] = eval_.apply(spacy_word_mask_to_spans, field = 'y_preds', axis = 1)
            out = eval_.pred_spans.to_list()
            
            to_submit(out, output_path='/home/burtenshaw/now/spans_toxic/predictions/submit/spans-pred_%s_%s.txt' % (METHOD_NAME, method.task_score))
                    


# %%
