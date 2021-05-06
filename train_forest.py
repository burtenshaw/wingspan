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
import json
from tqdm import *

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp

from models import to_ensemble
from utils import spacy_word_mask_to_spans, f1
from submit import align_predictions
from submit import to_submit

from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    # ipython convenience
    mgc = get_ipython().magic
    mgc(u'%load_ext tensorboard')
    mgc(u'%load_ext autoreload')
    mgc(u'%autoreload 2')
    METHOD_NAME = 'ROBERTA'
    LOG_DIR = "logs/" + METHOD_NAME
    os.chdir('/home/burtenshaw/now/spans_toxic')
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
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

import models
from utils import f1

tf.config.list_physical_devices(device_type='GPU')
fold = 'all'
data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
output_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)
save_path = os.path.join(output_dir, '%s.json' % (METHOD_NAME))
# word_ids_path = os.path.join(output_dir, '%s.bin' % ('token_bert' + '_word_ids'))
# model_path = os.path.join(output_dir, '%s.bin' % ('token_bert' + '_model'))

#%%

LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, fold)

print('logging at :', LOG_DIR)
print('data_dir : ', data_dir)
print('output_dir : ', output_dir)


train = pd.read_json(os.path.join(data_dir, "train.json"))
val = pd.read_json(os.path.join(data_dir, "val.json"))
test = pd.read_json(os.path.join(data_dir, "test.json"))


train.drop(columns = ['baseline_word_mask'], inplace = True)
val.drop(columns = ['baseline_word_mask'], inplace = True)
test.drop(columns = ['baseline_word_mask'], inplace = True)

print('train : ', train.shape)
print('val : ', val.shape)
print('test : ', test.shape)

#%%

from transformers import AlbertTokenizerFast
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class Forest(models.TokenBert):

    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen= maxlen
        self.tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')       
        
        self.methods = ['ELECTRA', 'ROBERTA', 'ALBERT']

        self.add_bert_sequences()
        self.get_other_predictions()

        _labels = [0,1,2]
        self.labels = {k:v for k,v in zip(_labels, list(to_categorical(_labels)))}
        self.weights = {0:1, 1:1, 2:1}

    def get_data(self):

        self.X_train = [np.vstack(self.train.input_ids.values).astype(float),
                        np.vstack(self.train.attn_mask.values).astype(float)]

        self.X_val =   [np.vstack(self.val.input_ids.values).astype(float),
                        np.vstack(self.val.attn_mask.values).astype(float)]

        self.X_test  = [np.vstack(self.test.input_ids.values).astype(float),
                        np.vstack(self.test.attn_mask.values).astype(float)]

        self.X_train.append(np.hstack(self.train.apply(self.make_other_labels, \
                            field = 'electra_a_pred', axis = 1).values))
        self.X_val.append(np.hstack(self.val.apply(self.make_other_labels, \
                            field = 'electra_a_pred', axis = 1).values))
        self.X_test.append(np.hstack(self.test.apply(self.make_other_labels, \
                            field = 'electra_a_pred', axis = 1).values))

        self.X_train.append(np.hstack(self.train.apply(self.make_other_labels, \
                            field = 'roberta_a_pred', axis = 1).values))
        self.X_val.append(np.hstack(self.val.apply(self.make_other_labels, \
                            field = 'roberta_a_pred', axis = 1).values))
        self.X_test.append(np.hstack(self.test.apply(self.make_other_labels, \
                            field = 'roberta_a_pred', axis = 1).values))

        self.X_train.append(np.hstack(self.train.apply(self.make_other_labels, \
                            field = 'albert_a_pred', axis = 1).values))
        self.X_val.append(np.hstack(self.val.apply(self.make_other_labels, \
                            field = 'albert_a_pred', axis = 1).values))
        self.X_test.append(np.hstack(self.test.apply(self.make_other_labels, \
                            field = 'albert_a_pred', axis = 1).values))

        self.y_train  = np.hstack(self.train.apply(self.make_target_labels, axis=1).values)
        self.y_val  = np.hstack(self.val.apply(self.make_target_labels, axis=1).values)
        self.y_test  = np.hstack(self.test.apply(self.make_target_labels, axis=1).values)
        
        # self.train_weights = self.get_class_weights(self.y_train).astype(float)

        return None


    def make_other_labels(self, row, field = 'word_mask'):
        labels = [row[field][word_idx] if word_idx != None else self.labels[2] for word_idx in row.word_ids]        
        sequences = np.vstack(labels).T
        return sequences 

    def align(self, y_pred, word_ids):

        a_preds = np.zeros((self.maxlen*2,3))
        
        for p, w in zip(y_pred,word_ids):
            if w == None:
                continue
            else:
                a_preds[w] = p
    
        return a_preds

    def get_best_results(self, pred_dir):

        prediction_files = os.listdir(pred_dir)

        # print(prediction_files)

        # print(self.methods)

        bestest = []

        for m in self.methods:
            method_preds = [[float(f.strip('%s_' % m).strip('.json')), f] \
                    for f in prediction_files if m in f]
            method_preds.sort(key=lambda x: x[0])

            bestest.append(method_preds[-1][1])

        self.bestest = bestest

        print(bestest)
        
        df = pd.concat([pd.read_json(os.path.join(pred_dir, filename))\
            .add_prefix('%s_' % filename.split('_')[0].lower()) \
            for filename in bestest
            ], axis = 1)

        for m in self.methods:
            m = m.lower()
            df['%s_a_pred' % m] = df.apply(lambda row : self.align(\
                                    row['%s_y_pred' % m], row['%s_word_ids' % m])\
                                        , axis =1)

        return df

    def get_other_predictions(self):

        fold_dir = [str(n) for n in range(5)]
        self.other_models = pd.DataFrame()

        for fold in fold_dir:
            data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
            pred_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)

            test = pd.read_json(os.path.join(data_dir, 'test.json'))

            df = self.get_best_results(pred_dir)
            
            df.index = test.index

            self.other_models = pd.concat([self.other_models, df],
                                        ignore_index = False, axis= 0)
        
        self.train['electra_a_pred'] = self.other_models.\
                                        loc[self.train.index].electra_a_pred
        self.val['electra_a_pred'] = self.other_models.\
                                        loc[self.val.index].electra_a_pred
        self.test['electra_a_pred'] = self.other_models.\
                                        loc[self.test.index].electra_a_pred

        self.train['roberta_a_pred'] = self.other_models.\
                                        loc[self.train.index].roberta_a_pred
        self.val['roberta_a_pred'] = self.other_models.\
                                        loc[self.val.index].roberta_a_pred
        self.test['roberta_a_pred'] = self.other_models.\
                                        loc[self.test.index].roberta_a_pred

        self.train['albert_a_pred'] = self.other_models.\
                                        loc[self.train.index].albert_a_pred
        self.val['albert_a_pred'] = self.other_models.\
                                        loc[self.val.index].albert_a_pred
        self.test['albert_a_pred'] = self.other_models.\
                                        loc[self.test.index].albert_a_pred
    def run(self, return_model = False): 
        pass
        
        
#%%
method = Forest(train, val, test)
method.get_data()
#%%

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV

X = np.vstack(method.X_train[3:]).T
y = np.argmax(method.y_train, 0)
#%%
max_features=None
class_weight={0:1,1:2,2:1}
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X, y)
#%%

X_test = np.vstack(method.X_test[3:]).T
y_test = np.argmax(method.y_test, 0)
y_pred = clf.predict(X_test)

alone = pd.DataFrame(classification_report(np.argmax(method.X_test[-1],0), y_test, output_dict=True))
ensemble = pd.DataFrame(classification_report(y_pred, y_test, output_dict=True))


# , sample_weight = compute_sample_weight({0:1,1:2,2:1}, y)
# n_estimators = [100, 300, 500, 800, 1200]
# max_depth = [5, 8, 15, 25, 30]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 

# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)

# gridF = GridSearchCV(clf, hyperF, cv = 3, verbose = 1, 
#                       n_jobs = -1)

# clf = gridF.fit(method.X_train, y)

# #%%
# cheap_pred = np.dstack(df.electra_y_pred.apply(lambda x : np.vstack(np.array(x).T)).values).T \
#     + np.dstack(df.roberta_y_pred.apply(lambda x : np.vstack(np.array(x).T)).values).T \
#         + np.dstack(df.albert_y_pred.apply(lambda x : np.vstack(np.array(x).T)).values).T

# cheap_pred = np.argmax(cheap_pred, 2)

#%%
''' majority voting '''


# cheap_pred = np.dstack([np.vstack(df.electra_y_pred.apply(lambda x : np.argmax(np.vstack(np.array(x)), 1)).values.T), \
#                         np.vstack(df.roberta_y_pred.apply(lambda x : np.argmax(np.vstack(np.array(x)), 1)).values.T), \
#                         np.vstack(df.albert_y_pred.apply(lambda x : np.argmax(np.vstack(np.array(x)), 1)).values.T)])

majority = lambda x : np.argmax(np.bincount(x))
on_models = lambda x : np.apply_along_axis(lambda x : np.apply_along_axis(majority, 0, x),0,x)
majority_voted = np.apply_along_axis(on_models, 0, cheap_pred)

#%%

# def make_features(row):
#     return list(np.vstack([np.argmax(np.array(row.electra_y_pred), 1), 
#                       np.argmax(np.array(row.roberta_y_pred), 1), 
#                       np.argmax(np.array(row.albert_y_pred), 1)]).T)

# df['forest_features'] = df.apply(make_features, axis = 1)
# do_prediction = lambda x : np.argmax(clf.predict(x), 1)
# df['forest_predictions'] = df.forest_features.apply(clf.predict)
# pred = df.forest_predictions.values.tolist()

pred = majority_voted

w_id_predictions, aligned_predictions = align_predictions(pred, method.test.word_ids, method.test.attn_mask)

true = method.test.word_mask.apply(lambda x : np.where(np.array(x)== 1)[0]).to_list()
pred = w_id_predictions

all_fs = [f1(p,t) for t,p in zip(true,pred)]
print('Word level f1 : ', np.mean(all_fs))

#%%
test['y_preds'] = aligned_predictions
test['pred_spans'] = test.apply(spacy_word_mask_to_spans, field = 'y_preds', axis = 1)

test['f1_score'] = test.apply(lambda row : f1(row.pred_spans, row.spans) , axis = 1)
print('span f1 : ', test.f1_score.mean())
#%%
# to_ensemble(y_pred, method, save_path)

# # %%
# eval_ = pd.read_pickle(os.path.join('/home/burtenshaw/now/spans_toxic/data', "eval.bin"))
# X_eval = method.evaluation_data(eval_)
# y_eval = model.predict(X_eval)
# y_eval = list(np.argmax(y_eval[:,:,:], 2))
# _, eval_['y_preds'] = align_predictions(y_eval, eval_.word_ids, eval_.attn_mask)
# eval_['pred_spans'] = eval_.apply(spacy_word_mask_to_spans, field = 'y_preds', axis = 1)
# out = eval_.pred_spans.to_list()
# #%%
# to_submit(out, output_path='/home/burtenshaw/now/spans_toxic/predictions/submit/spans-pred.txt')


# # %%
