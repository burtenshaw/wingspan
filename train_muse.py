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

max_len = 200

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
data = pd.read_pickle(data_dir + "train.bin")
data = data.loc[data.word_mask.apply(np.array).apply(sum) > 1]
X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(data.index, data.index, test_size=0.1, random_state=2018)
data = make_word_level_df(data)

rdf = data.loc[X_test_index]
rdf.columns = pd.MultiIndex.from_product([['base'],rdf.columns])
#%%
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

print(embeddings)

# The following are example embedding output of 512 dimensions per sentence
# Embedding for: The quick brown fox jumps over the lazy dog.
# [-0.03133016 -0.06338634 -0.01607501, ...]
# Embedding for: I am a sentence for which I would like to get its embedding.
# [0.05080863 -0.0165243   0.01573782, ...]

embedding_matrix, num_tokens, vectorizer = muse_embedding(data.loc[X_train_index].text.values)

x_s_train = vectorizer(np.array([[s] for s in data.loc[X_train_index].text])).numpy()
x_s_val = vectorizer(np.array([[s] for s in data.loc[X_test_index].text])).numpy()

x_w_train = vectorizer(np.array([[s] for s in data.loc[X_train_index].word])).numpy()
x_w_val = vectorizer(np.array([[s] for s in data.loc[X_test_index].word])).numpy()

y_train = np.array(data.loc[y_train_index].label)
y_val = np.array(data.loc[y_train_index].label)

#%%
def create_model(n_layers = 1, n_nodes=100, activation='relu', lr = 0.1, batch_size= None, dropout=0.1):  

    sentence_input = tf.keras.Input(shape=(None,), dtype="int64")
    word_input = tf.keras.Input(shape=(None,), dtype="int64")

    sentence_features = layers.Embedding(num_tokens, 300, 
                                    weights=[embedding_matrix], 
                                    trainable=False)(sentence_input)

    word_features = layers.Embedding(num_tokens, 300, 
                                    weights=[embedding_matrix], 
                                    trainable=False)(word_input)

    sentence_features = layers.Bidirectional(layers.LSTM(150))(sentence_features)
    word_features = layers.Bidirectional(layers.LSTM(150))(word_features)

    x = layers.concatenate([sentence_features, word_features])

    x = layers.Dense(300, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(150, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(75, activation=activation)(x)

    output = layers.Dense(1, activation = 'sigmoid', name="priority")(x)

    model = tf.keras.Model(
        inputs=[sentence_input, word_input],
        outputs=[output],
    )


    bin_acc = tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5
    )

    opt = Adam(lr = lr)

    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = [bin_acc])

    return model

#%% TRAIN

# Define the parameters to try out
params = {
          'activation': ['relu', 'tanh'], 
          'batch_size': [32, 64, 128], 
          'lr': [0.1, 0.01, 0.001], 
          'dropout' : [0.1, 0.2, 0.4]
          }


monitor_val_acc = EarlyStopping(monitor='val_binary_accuracy', 
                       patience=2)

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