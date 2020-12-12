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

max_len = 200

data_dir = '/home/burtenshaw/now/spans_toxic/data/'

data = pd.read_pickle(data_dir + "train.bin")

data = make_word_level_df(data)

# %%


X = data[['text', 'word']].values
y = data.label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2018)

X_s_mask,X_s_ids = bert_prep([x[0] for x in X_train], max_len=200)
X_w_mask,X_w_ids = bert_prep([x[1] for x in X_train], max_len=200)

#%%


# def build_bert(input_dim = 200, output_dim=6, dropout=0.2):
    
#     s_input_1 = tf.keras.Input(shape = (input_dim) , dtype=np.int32)
#     s_input_2 = tf.keras.Input(shape = (input_dim) , dtype=np.int32)
#     s_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
#     s_model  = s_model([s_input_1 , s_input_2], training = True )

#     w_input_1 = tf.keras.Input(shape = (1) , dtype=np.int32)
#     w_input_2 = tf.keras.Input(shape = (1) , dtype=np.int32)
#     w_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
#     w_model  = w_model([w_input_1 , w_input_2], training = True )
    
#     combined = layers.Concatenate()([s_model[0], w_model[0]])

#     answer = layers.Dense(2, activation=tf.nn.sigmoid)(combined)
#     answer = layers.Dense(1, activation=tf.nn.sigmoid)(answer)

#     model = tf.keras.Model(inputs = [[s_input_1, s_input_], [w_input_1, w_input_2] ] , outputs = [answer])

#     model.summary()

#     model.compile(optimizer = Adam(lr = 3e-5),
#                   loss = tf.keras.losses.binary_crossentropy,
#                   metrics = ['accuracy'])
#     return model



bert_model = build_bert(output_dim=1)


#%%
bert_model.fit([X_ids , X_mask] , y_train,
            validation_split=0.1,
            batch_size = 12, 
            epochs=3,
)

X_mask,X_ids = bert_prep(X_test)
results= bert_model.predict([X_ids,X_mask])

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
