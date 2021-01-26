#%%
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from ast import literal_eval
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

df = pd.read_pickle("/home/burtenshaw/now/spans_toxic/data/toxic_train_LEXICAL.bin")


#%%

X = df.lexical_prediction.to_list()
y = df.target_mask.to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

max_len = 200
#%% 
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=   , activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

metrics = [
    tf.keras.metrics.AUC(curve='ROC'),
    tf.keras.metrics.AUC(curve='PR')

]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
model.fit(X_train, y_train, epochs=200, verbose=1)

y_pred = model.predict(X_test)

# scores = []

# for p,t in zip(y_pred,y_test):
#     scores.append(f1_score(t, p, average='macro'))

# print(np.mean(scores))
