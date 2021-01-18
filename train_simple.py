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
    METHOD_NAME = 'dev/token_bert'
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

import models
from utils import f1

tf.config.list_physical_devices(device_type='GPU')
fold = 'all'
data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
output_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)
save_path = os.path.join(output_dir, '%s.bin' % (METHOD_NAME))
word_ids_path = os.path.join(output_dir, '%s.bin' % ('token_bert' + '_word_ids'))
model_path = os.path.join(output_dir, '%s.bin' % ('token_bert' + '_model'))

#%%
LOG_DIR = "logs/%s/%s/" % (METHOD_NAME, fold)

print('logging at :', LOG_DIR)
print('data_dir : ', data_dir)
print('output_dir : ', output_dir)


train = pd.read_pickle(os.path.join(data_dir, "train.bin"))
val = pd.read_pickle(os.path.join(data_dir, "val.bin"))
test = pd.read_pickle(os.path.join(data_dir, "test.bin"))


# with open('data/hate_dataset.json', 'r') as f:
#     data = json.load(f)

# hate = pd.DataFrame(data).T[['post_tokens', 'rationales']]\
#     .explode('rationales').reset_index().dropna().drop(columns=['index'])\
#         .rename(columns = {'post_tokens' : 'tokens' , 'rationales' : 'word_mask'})

# hate['text']= hate.tokens.apply(lambda x : ' '.join(x)) 

# train = pd.concat([train,val,test[:-200], hate])
# val = test[-200:-100]

# test = test[-100:]

print('train : ', train.shape)
print('val : ', val.shape)
print('test : ', test.shape)

method = models.TokenBert
callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

hparams = { 
            'activation' : 'relu',
            'batch_size' : 8,
            'lr' : 0.0000001,
            'dropout' : 0.24,
            'n_layers' : 1,
            'model_scale' : 1, 
            'epochs' : 2, 
            'neg_weight' : 6,
            'pos_weight' : 0.6,
        }

method = method(train, val, test)
method.callbacks = callbacks
method.hparams = hparams

with open(word_ids_path, 'wb') as f :
    np.save(f, method.test.word_ids)

#%%
train_samples = method.get_data()

#%%

model = method.run(data = train_samples, return_model = True)

y_pred = model.predict(method.X_test)

#%%
with open(save_path, 'wb') as f :
    np.save(f, y_pred)

model.save(model_path)

#%%