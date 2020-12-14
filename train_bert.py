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


tf.config.list_physical_devices(device_type='GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

max_len = 200

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
data = pd.read_pickle(data_dir + "train.bin")
#%%

data = data.loc[data.word_mask.apply(np.array).apply(sum) > 1]
X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(data.index, data.index, test_size=0.1, random_state=2018)
data = make_word_level_df(data)

#%%

embedding_matrix, num_tokens, vectorizer = muse_embedding(data.loc[X_train_index].text.values)

x_s_train = vectorizer(np.array([[s] for s in data.loc[X_train_index].text])).numpy()
x_s_val = vectorizer(np.array([[s] for s in data.loc[X_test_index].text])).numpy()

x_w_train = vectorizer(np.array([[s] for s in data.loc[X_train_index].word])).numpy()
x_w_val = vectorizer(np.array([[s] for s in data.loc[X_test_index].word])).numpy()

y_train = np.array(data.loc[y_train_index].label)
y_val = np.array(data.loc[y_train_index].label)

#%%


def dual_bert():

    '''
    https://github.com/cerlymarco/MEDIUM_NoteBook/blob/master/Siamese_Dual_BERT/Siamese_Dual_BERT.ipynb
    '''
    
    opt = Adam(learning_rate=2e-5)
    
    id1 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    id2 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    mask1 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    mask2 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    atn1 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    atn2 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    config = BertConfig() 
    config.output_hidden_states = False # Set to True to obtain hidden states
    bert_model1 = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    bert_model2 = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    embedding1 = bert_model1(id1, attention_mask=mask1, token_type_ids=atn1)[0]
    embedding2 = bert_model2(id2, attention_mask=mask2, token_type_ids=atn2)[0]
    
    x1 = GlobalAveragePooling1D()(embedding1)
    x2 = GlobalAveragePooling1D()(embedding2)
    
    x = Concatenate()([x1, x2])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(len(map_label), activation='softmax')(x)

    model = Model(inputs=[id1, mask1, atn1, id2, mask2, atn2], outputs=out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    
    return model



model = tf.keras.Model(
    inputs=[title_input, body_input],
    outputs=[priority_pred],
)


bin_acc = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [bin_acc])

#%% TRAIN
monitor_val_acc = EarlyStopping(monitor='val_binary_accuracy', 
                       patience=2)

history = model.fit([x_s_train, x_w_train] , y_train,
            validation_split=0.3,
            batch_size = 36, 
            epochs=8,
            callbacks=[monitor_val_acc])

results= model.predict([x_s_val,x_w_val])
#%%
rdf = data.loc[X_test_index]
rdf.columns = pd.MultiIndex.from_product([rdf.columns, ['base']])
rdf['muse','pred'] = results.flatten().astype(float)

component_models = [('muse', 0.5)]
r = EvalResults(component_models, rdf)
# rdf.to_pickle('data/muse_multi_input.bin')
#%%