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
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from transformers import AlbertTokenizerFast, TFAlbertForTokenClassification

class AlbertAndFriends(models.TokenBert):


    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen= maxlen
        self.tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
        self.model = TFAlbertForTokenClassification.from_pretrained('albert-base-v2')        
        
        self.methods = ['ELECTRA', 'ROBERTA', 'ALBERT']

        self.add_bert_sequences()
        self.get_other_predictions()

        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        ]
        
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

        self.X_train.append(np.dstack(self.train.apply(self.make_other_labels, \
                            field = 'electra_a_pred', axis = 1).values).T)
        self.X_val.append(np.dstack(self.val.apply(self.make_other_labels, \
                            field = 'electra_a_pred', axis = 1).values).T)
        self.X_test.append(np.dstack(self.test.apply(self.make_other_labels, \
                            field = 'electra_a_pred', axis = 1).values).T)

        self.X_train.append(np.dstack(self.train.apply(self.make_other_labels, \
                            field = 'roberta_a_pred', axis = 1).values).T)
        self.X_val.append(np.dstack(self.val.apply(self.make_other_labels, \
                            field = 'roberta_a_pred', axis = 1).values).T)
        self.X_test.append(np.dstack(self.test.apply(self.make_other_labels, \
                            field = 'roberta_a_pred', axis = 1).values).T)

        self.X_train.append(np.dstack(self.train.apply(self.make_other_labels, \
                            field = 'albert_a_pred', axis = 1).values).T)
        self.X_val.append(np.dstack(self.val.apply(self.make_other_labels, \
                            field = 'albert_a_pred', axis = 1).values).T)
        self.X_test.append(np.dstack(self.test.apply(self.make_other_labels, \
                            field = 'albert_a_pred', axis = 1).values).T)

        self.y_train  = np.dstack(self.train.apply(self.make_target_labels, axis=1).values).T
        self.y_val  = np.dstack(self.val.apply(self.make_target_labels, axis=1).values).T
        self.y_test  = np.dstack(self.test.apply(self.make_target_labels, axis=1).values).T
        
        self.train_weights = self.get_class_weights(self.y_train).astype(float)

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

        print(prediction_files)

        print(self.methods)

        bestest = []

        for m in self.methods:
            method_preds = [[float(f.strip('%s_' % m).strip('.json')), f] \
                    for f in prediction_files if m in f]
            method_preds.sort(key=lambda x: x[0])

            bestest.append(method_preds[-1][1])

        self.bestest = bestest
        
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

    def lr_scheduler(self, epoch, lr):
        return lr * self.hparams['epsilon']

    def build_model(self):
            
        hp = self.hparams

        ids = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        attn_mask = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        electra = tf.keras.layers.Input((self.maxlen, 3), dtype=tf.float32)
        roberta = tf.keras.layers.Input((self.maxlen, 3), dtype=tf.float32)
        albert = tf.keras.layers.Input((self.maxlen, 3), dtype=tf.float32)
    
        layer = self.model(input_ids = ids,
                           attention_mask = attn_mask
                          )[0]

        layer = tf.keras.layers.Dense(3)(layer)

        electra_nn = tf.convert_to_tensor(electra) + 1
        roberta_nn = tf.convert_to_tensor(roberta) + 1
        albert_nn = tf.convert_to_tensor(albert) + 1

        # layer = tf.keras.layers.Concatenate(-1)([electra_nn, 
        #                                         roberta_nn, 
        #                                         albert_nn, 
        #                                         layer
        #                                         ])

        layer = tf.keras.layers.Dense(3)(layer * electra_nn)
        layer = tf.keras.layers.Dropout(.2)(layer)
        layer = tf.keras.layers.Dense(3)(layer * albert_nn)
        layer = tf.keras.layers.Dropout(.2)(layer)
        layer = tf.keras.layers.Dense(3)(layer * roberta_nn)
        layer = tf.keras.layers.Dropout(.2)(layer)

        out = tf.keras.layers.Dense(3, activation='softmax')(layer) 

        model = tf.keras.Model( inputs=[
                                        ids,
                                        attn_mask,
                                        electra,
                                        roberta,
                                        albert
                                        ], 
                                outputs=out)

        
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = hp['lr']), 
                    loss = 'categorical_crossentropy', 
                    metrics = self.metrics)
        
        model.summary()

        return model

    def run(self, data, return_model = False): 
        
        model = self.build_model()

        self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1))

        model.fit(  self.X_train, 
                    self.y_train,
                    batch_size=self.hparams['batch_size'],
                    validation_data=(self.X_val, self.y_val),
                    epochs=self.hparams['epochs'],
                    verbose = 1,
                    callbacks= self.callbacks,
                    sample_weight = self.train_weights)
                    
        self.y_pred = model.predict(self.X_test)
        task_score = self.task_results(self.y_test, self.y_pred)

        if return_model:
            return model
        else:
            scores = model.evaluate(self.X_test, self.y_test, return_dict = True)
            scores['task_f1'] = task_score
            return scores


#%%

method = AlbertAndFriends

callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

method.callbacks = callbacks

hparams = {
            'batch_size' : 8,
            'lr' : 0.00001,
            'dropout' : .1,
            'epochs' : 4,
            'neg_weight' : 1,
            'pos_weight' : 1.2,
            'pad_weight' : 1,
            'epsilon' : .9,
            'nodes' : 12,
            'n_layers' : 3,
        }

method.hparams = hparams
method = method(train, val, test)
train_samples = method.get_data()

# #%%
# base = pd.DataFrame(np.vstack([np.hstack(method.other_models[col].values) \
#         for col in method.other_models.columns if 'word_ids' in col])).T\
#             .applymap(lambda x : -1 if x == None else x)

# look = pd.DataFrame(np.vstack([np.argmax(x,2).flatten() \
#         for x in method.X_train[2:] ]).T)

model = method.run(data = train_samples, return_model = True)

#%%
y_pred = model.predict(method.X_test)

#%%


#%%

from models import to_ensemble
from utils import spacy_word_mask_to_spans, f1
from submit import align_predictions
from submit import to_submit

# y_pred = np.dstack([y_pred[:,:,0], y_pred[:,:,1] * 2, y_pred[:,:,2]])
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
