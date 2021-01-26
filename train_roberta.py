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
word_ids_path = os.path.join(output_dir, '%s.bin' % ('token_bert' + '_word_ids'))
model_path = os.path.join(output_dir, '%s.bin' % ('token_bert' + '_model'))

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

# eval_ = pd.read_pickle(os.path.join(data_dir, "eval.bin"))


# with open('data/hate_dataset.json', 'r') as f:
#     data = json.load(f)

# hate = pd.DataFrame(data).T[['post_tokens', 'rationales']]\
#     .explode('rationales').reset_index().dropna().drop(columns=['index'])\
#         .rename(columns = {'post_tokens' : 'tokens' , 'rationales' : 'word_mask'})

# hate['text']= hate.tokens.apply(lambda x : ' '.join(x)) 

# train = pd.concat([train,val,hate])
# val = test[:800]
# test = test[800:]

print('train : ', train.shape)
print('val : ', val.shape)
print('test : ', test.shape)

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import transformers

from transformers import RobertaTokenizerFast, TFRobertaForTokenClassification

class Roberta(models.Roberta):


    def run(self, data, return_model = False): 
        
        hp = self.hparams
        
        ids = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        # tok_types = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        attn_mask = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
    
        layer = self.bert_model(input_ids = ids,
                                attention_mask = attn_mask,
                                # token_type_ids = tok_types
                                )[0]

        for n in range(hp['n_layers']):
            layer = layers.Dense(hp['nodes'][n])(layer)
            layer = layers.Dropout(hp['dropout'])(layer)

        out = layers.Dense(3, activation='softmax')(layer) 

        model = tf.keras.Model( inputs=[ids, 
                                        # tok_types, 
                                        attn_mask,
                                        # class_weights
                                        ], 
                                outputs=out)

        

        model.summary()

        opt = Adam(lr = hp['lr'])
        model.compile(optimizer = opt, 
                    loss = 'categorical_crossentropy', 
                    metrics = self.metrics)
                    
        model.summary()

        model.fit(  self.X_train , 
                    self.y_train,
                    batch_size=hp['batch_size'],
                    validation_data=(self.X_val, self.y_val),
                    epochs=hp['epochs'],
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

method = Roberta
callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]


hparams = { 
            'activation' : 'relu',
            'batch_size' : 8,
            'lr' : 0.00001,
            'dropout' : 0.2,
            'n_layers' : 3,
            'epochs' : 2, 
            'neg_weight' : 1.0,
            'pos_weight' : 1.2,
            'pad_weight' : 1.0,
            'nodes' : [12,6,3]
        }

method = method(train, val, test)
method.callbacks = callbacks
method.hparams = hparams


train_samples = method.get_data()
model = method.run(data = train_samples, return_model = True)
y_pred = model.predict(method.X_test)

#%%

from models import to_ensemble





to_ensemble(y_pred, method, save_path)

#%%

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

# %%
eval_ = pd.read_pickle(os.path.join(data_dir, "eval.bin"))
X_eval = method.evaluation_data(eval_)
y_eval = model.predict(X_eval)
y_eval = list(np.argmax(y_eval[:,:,:], 2))
_, eval_['y_preds'] = align_predictions(y_eval, eval_.word_ids, eval_.attn_mask)
eval_['pred_spans'] = eval_.apply(spacy_word_mask_to_spans, field = 'y_preds', axis = 1)
out = eval_.pred_spans.to_list()
#%%
to_submit(out, output_path='/home/burtenshaw/now/spans_toxic/predictions/submit/spans-pred.txt')
# %%
