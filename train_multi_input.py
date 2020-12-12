# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification
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

max_len = 200

data_dir = '/home/burtenshaw/now/spans_toxic/data/'

data = pd.read_pickle(data_dir + "train.bin")

data = make_word_level_df(data)
#%%
pos = data[data.label == 1]
neg = data[data.label == 0][:pos.shape[0]]
data = pd.concat([pos, neg])

# %%
X = data[['text', 'word']]
y = data.label.values


embedding_matrix, num_tokens, vectorizer = muse_embedding(X.text.values)

def muse_prep(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2018)
    x_train = vectorizer(np.array([[s] for s in X_train])).numpy()
    x_val = vectorizer(np.array([[s] for s in X_test])).numpy()
    y_train = np.array(y_train)
    y_val = np.array(y_test)
    return x_train, x_val, y_train, y_val

x_s_train, x_s_val, y_train, y_val = muse_prep(X.text, y)
x_w_train , x_w_val , __ , _ = muse_prep(X.word, y)

#%%

num_departments = 1  

title_input = tf.keras.Input(shape=(None,), dtype="int64")
body_input = tf.keras.Input(shape=(None,), dtype="int64")

title_features = layers.Embedding(num_tokens, 300, 
                                weights=[embedding_matrix], 
                                trainable=False)(title_input)

body_features = layers.Embedding(num_tokens, 300, 
                                weights=[embedding_matrix], 
                                trainable=False)(body_input)

title_features = layers.Bidirectional(layers.LSTM(150, return_sequences=True))(title_features)
body_features = layers.Bidirectional(layers.LSTM(150, return_sequences=True))(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features])
x = layers.Dense(300, activation='relu')(x)
x = layers.Dense(150, activation='relu')(x)
x = layers.Dense(50, activation='relu')(x)

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, activation = 'sigmoid', name="priority")(x)


model = tf.keras.Model(
    inputs=[title_input, body_input],
    outputs=[priority_pred],
)

#%%
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit([x_s_train, x_w_train] , y_train,
            validation_split=0.3,
            batch_size = 36, 
            epochs=8,
)

#%%

results= model.predict([x_s_train,x_w_train])

#%%

bert_e = EvalResults(results, y_test, roc = True, token_type='word')

#%%
embedding_matrix, num_tokens, vectorizer  = muse_embedding(X_train)
x_train, x_val, y_train, y_val  = muse_prep(X_train, X_test, y_train, y_test, vectorizer)

#%%
muse_model=build_muse_lstm(embedding_matrix, num_tokens, output_dim=max_tags)
muse_model.fit(x_train, y_train, batch_size=128, epochs=8, validation_split=0.1)

#%%
results = muse_model.predict(x_val)

#%%
muse_e = EvalResults(results, y_test, roc = True)

#%%% UniversalEmbedding
