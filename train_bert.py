# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification, BertConfig
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_curve

from tensorflow.keras.utils import to_categorical
os.chdir('/home/burtenshaw/now/spans_toxic')

%load_ext autoreload
%autoreload 2

from results import EvalResults
from utils import *
from models import *

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re
import random

tf.config.list_physical_devices(device_type='GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

max_len = 200

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
data = pd.read_pickle(data_dir + "train.bin")

data = data.loc[data.word_mask.apply(np.array).apply(sum) > 1]
X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(data.index, data.index, test_size=0.1, random_state=2018)
X_train_index, X_val_index, y_train_index, y_val_index = train_test_split(X_train_index, y_train_index, test_size=0.1, random_state=2018)
data = make_word_level_df(data, max_len = 128)

rdf = data.loc[X_test_index]
rdf.columns = pd.MultiIndex.from_product([['base'],rdf.columns])
#%%
X_train_id, _, X_train_attn = bert_prep(data.loc[y_train_index].text.to_list(), max_len = 128)
X_val_id, _, X_val_attn = bert_prep(data.loc[y_val_index].text.to_list(), max_len = 128)
X_test_id, _, X_test_attn = bert_prep(data.loc[y_test_index].text.to_list(), max_len = 128)

X_train_mask = np.vstack(data.loc[y_train_index].x_word_mask.values)
X_val_mask = np.vstack(data.loc[y_val_index].x_word_mask.values)
X_test_mask = np.vstack(data.loc[y_test_index].x_word_mask.values)

y_train = np.array(data.loc[y_train_index].label)
y_val = np.array(data.loc[y_val_index].label)
y_test = np.array(data.loc[y_test_index].label)

#%%
def single_bert(n_layers = 1, lstm_nodes = 150, nn_nodes = 64, activation = 'relu', lr = 0.1, batch_size = None, dropout = 0.1):  
    
    opt = Adam(learning_rate=2e-5)
    
    id1 = tf.keras.layers.Input((128,), dtype=tf.int32)
    mask1 = tf.keras.layers.Input((128,), dtype=tf.int32)
    atn1 = tf.keras.layers.Input((128,), dtype=tf.int32)
    
    config = BertConfig() 
    config.output_hidden_states = False # Set to True to obtain hidden states
    bert_model1 = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    embedding_layer = bert_model1(id1, attention_mask=atn1, token_type_ids=mask1)[0]

    bi_lstm_1 = layers.Bidirectional(layers.LSTM(lstm_nodes, return_sequences=True))(embedding_layer)
    bi_lstm_2 = layers.Bidirectional(layers.LSTM(lstm_nodes))(bi_lstm_1)
    # x = tf.keras.layers.GlobalAveragePooling1D()(embedding1)
    
    x = tf.keras.layers.Dense(nn_nodes, activation=activation)(bi_lstm_2)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[id1, mask1, atn1], outputs=out)
    bin_acc = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
    )
    opt = Adam(lr = lr)

    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = [bin_acc])
    return model

#%% TRAIN

monitor_val_acc = EarlyStopping(monitor='val_binary_accuracy', 
                       patience=2)

# Define the parameters to try out
params = {
          'activation': ['relu', 'tanh'], 
          'batch_size': [8, 16], 
          'lr': [0.1, 0.01, 0.001], 
          'dropout' : [0.1, 0.2, 0.4]
          }


monitor_val_acc = EarlyStopping(monitor='val_binary_accuracy', 
                       patience=2)

used_params = {}
history_log = []

for n in range(5):
    p = {k : random.choice(params[k]) for k in params.keys()}
    used_params[n] = p
    print(p)
    model = single_bert(**p)
    history = model.fit([X_train_id,X_train_mask, X_train_attn] , y_train,
                validation_data=([X_val_id,X_val_mask, X_val_attn], y_val),
                batch_size = 16, 
                epochs=8,
                callbacks=[monitor_val_acc])

    results= model.predict([X_val_id, X_val_mask, X_val_attn])

    history_log.append(history)

    rdf['bert_%s' % n,'pred'] = results.flatten().astype(float)



#%%
rdf = data.loc[X_test_index]
rdf.columns = pd.MultiIndex.from_product([rdf.columns, ['base']])
rdf['muse','pred'] = results.flatten().astype(float)

component_models = [('muse', 0.5)]
r = EvalResults(component_models, rdf)
# rdf.to_pickle('data/muse_multi_input.bin')
#%%