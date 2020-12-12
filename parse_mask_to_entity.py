#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import f1_score
os.chdir('/home/burtenshaw/now/spans_toxic')
from utils import *

tf.config.list_physical_devices(device_type='GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
data = pd.read_pickle(data_dir + "train.bin")

#%%
X = np.vstack(data['word_mask'].apply(np.array).apply(pad_mask, \
    max_len = data['word_mask'].apply(len).max()).values)

y = np.vstack(data['flat_cont_ranges'].apply(np.array).apply(pad_mask,\
     max_len = data['flat_cont_ranges'].apply(len).max()).apply(lambda x : x[:5]).values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2018)
#%%
model = Sequential()
model.add(tf.keras.Input(shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.metrics.AUC()])
# %%
model.fit(X_train, y_train, epochs= 10)
# %%
y_pred = model.predict(X_test)
np.argmax(to_categorical(np.asarray(y_pred)), -1)[:20]
# %%
f1_score(y_test, y_pred)
# %%
