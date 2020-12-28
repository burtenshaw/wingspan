#%%
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from pickle import load
import os
os.chdir('/home/burtenshaw/now/spans_toxic')

%load_ext autoreload
%autoreload 2

from results import EvalResults
from utils import *
from models import *

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

tf.config.list_physical_devices(device_type='GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# %%
''' 

predict word label based on : 

- model prediction for each word in sentence
- model prediction for current word
- lexical toxicity for full sentence
- lexical toxicity for word 

'''

# categorical
ibm = pd.read_pickle('data/TRAIN_IBM_TOX.bin')


#task models
muse = pd.read_pickle('data/muse_multi_input.bin')
lex = pd.read_pickle('data/lexical_pred.bin')

muse['sentence_pred'] = muse.groupby(level=0).pred.apply(np.array).apply(pad_mask, max_len =239)
lex['sentence_pred'] = lex.groupby(level=0).lexical.apply(np.array).apply(pad_mask, max_len =239)
lex = lex[['sentence_pred','lexical','label']].loc[muse.index.drop_duplicates()].rename(columns={'lexical':'pred'})


df = pd.concat([muse,lex], axis= 1, keys = ['muse', 'lex'])
#%%
train_index, test_index = train_test_split(df.index.drop_duplicates(), test_size=0.3, random_state=2018)

'''
X = [SENTENCE_MASK, WORD_PREDICTION] * N_MODELS
Y = WORD_LABEL
'''
def make_X_y(df):

    X = np.hstack([ np.vstack(df.muse.sentence_pred.values),
                    np.vstack(df.muse.pred.values),
                    np.vstack(df.lex.sentence_pred.values),
                    np.vstack(df.lex.pred.values)])

    y = df.muse.label.values

    return X, y

X_train, y_train = make_X_y(df.loc[train_index]) 
X_test, y_test = make_X_y(df.loc[test_index])
# %% RANDO FOREST
clf = RandomForestClassifier(max_depth=40, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
#%%
ensemble_y_pred = clf.predict_proba(X_test)
#%%
results_df = df.loc[test_index]
results_df['ensemble', 'pred'] = np.amax(ensemble_y_pred, -1)
_data = pd.read_pickle("data/train.bin").loc[test_index]

component_models = [('muse', 0.5), ('lex', 0), ('ensemble', 0.01)]

r = EvalResults(component_models, results_df , params = {'muse' : {'lr' : 0, 'activation' : 'relu'}})
r.rdf# %%

# %%
