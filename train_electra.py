# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_hub as hub
import tensorflow_text as text  # Imports TF ops for preprocessing.

from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_curve
from sklearn.model_selection import RandomizedSearchCV, KFold

from tensorflow.keras.utils import to_categorical
os.chdir('/home/burtenshaw/now/spans_toxic')

%load_ext autoreload
%autoreload 2

from results import EvalResults
from utils import *
from models import *

import string
import re


tf.config.list_physical_devices(device_type='GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

max_len = 200

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
data = pd.read_pickle(data_dir + "train.bin")
data = data.loc[data.word_mask.apply(np.array).apply(sum) > 1]

X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(data.index, data.index, test_size=0.1, random_state=2018)
X_train_index, X_val_index, y_train_index, y_val_index = train_test_split(X_train_index, y_train_index, test_size=0.1, random_state=2018)


rdf = data.loc[X_test_index]
rdf.columns = pd.MultiIndex.from_product([['base'],rdf.columns])

# Load the BERT encoder and preprocessing models
# preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2')
# bert = hub.load('https://tfhub.dev/google/electra_large/2')
preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2')
bert = hub.load('https://tfhub.dev/google/electra_small/2')

# bert_inputs = preprocess(data.text.to_list())
# sequence_output = bert(bert_inputs)['sequence_output']
data = make_word_level_df(data)
#%%

x_s_train = tf.constant(data.loc[X_train_index].text.to_list())
x_w_train = tf.constant(data.loc[X_train_index].word.to_list())
y_train = np.array(data.loc[y_train_index].label)

x_s_val = tf.constant(data.loc[X_val_index].text.to_list())
x_w_val = tf.constant(data.loc[X_val_index].word.to_list())
y_val = np.array(data.loc[y_val_index].label)

x_s_test = tf.constant(data.loc[X_test_index].text.to_list())
x_w_test = tf.constant(data.loc[X_test_index].word.to_list())
y_test = np.array(data.loc[y_test_index].label)

#%%
VOCAB_SIZE=1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(data.word.values)

#%%

def build_classifier_model():
  
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  word_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='word')

  preprocessing_layer = hub.KerasLayer(preprocess, name='preprocessing')
  
  s_encoder_inputs = preprocessing_layer(text_input)
  w_encoder_inputs = encoder(word_input)

  w_embedding = tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True)(w_encoder_inputs)

  s_encoder = hub.KerasLayer(bert, trainable=True, name='BERT_encoder')
  s_outputs = s_encoder(s_encoder_inputs)
  s_outputs = s_outputs['pooled_output']

  w_outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(w_embedding)
  w_outputs = tf.keras.layers.Dense(64, activation='relu')(w_outputs)
  
  outputs = layers.concatenate([s_outputs, w_outputs])

  net = tf.keras.layers.Dense(50, activation='relu')(outputs)
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  
  return tf.keras.Model([text_input, word_input], net)

model = build_classifier_model()
model.summary()
#%%

monitor_val_acc = EarlyStopping(monitor='val_binary_accuracy', patience=2)
bin_acc = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [bin_acc])

h = model.fit([x_s_train, x_w_train] , y_train,
            batch_size=8,
            validation_data=([x_s_val, x_w_val], y_val),
            epochs=8,
            callbacks=[monitor_val_acc])

results = model.predict([x_s_val,x_w_val])


#%% TRAIN

# Define the parameters to try out
params = {
          'activation': ['relu', 'tanh'], 
          'batch_size': [32, 64, 128], 
          'lr': [0.1, 0.01, 0.001], 
          'dropout' : [0.1, 0.2, 0.4]
          }



used_params = {}

for n in range(5):
    p = {k : random.choice(params[k]) for k in params.keys()}
    used_params[n] = p
    print(p)
    model = create_model(**p)
    h = model.fit([x_s_train, x_w_train] , y_train,
                batch_size=p['batch_size'],
                validation_split=0.3,
                epochs=8,
                callbacks=[monitor_val_acc])

    results = model.predict([x_s_val,x_w_val])

    rdf['muse_%s' % n,'pred'] = results.flatten().astype(float)

#%%

component_models = [('muse_%s' % n, 0.5) for n in used_params.keys()]

r = EvalResults(component_models, rdf)
# rdf.to_pickle('data/muse_multi_input.bin')
#%%