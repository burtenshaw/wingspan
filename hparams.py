#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import seaborn as sns
import json
import os

log_dir = 'easy_logs'
names = os.listdir(log_dir)
fs = [pd.read_json(os.path.join(log_dir, f)) for f in names]
for df, fname in zip(fs, names):
    info = fname.split('_')
    df['name'] = info[0]
    df['fold'] = info[1]
    df['time'] = info[2].split('.')[0]


# [c for c in logs.columns if 'sensitivity' not in c]
cols = [
    'batch_size',
#  'lr',
 'dropout',
 'epochs',
 'n_layers',
 'neg_weight',
 'pos_weight',
 'pad_weight',
#  'loss',
#  'categorical_accuracy',  
#  'auc',
'epsilon',
 'task_f1',
 'name',
 'fold',
 'time']

cols = df.columns

logs = pd.concat(fs, axis =0, names=names)[cols]
logs.index = list(range(logs.shape[0]))


now = datetime.now()
yesterday = now - timedelta(days=1)

def get_best(method, fold, since = yesterday, n = 20):

    df = logs.loc[(logs.name == method) \
                & (logs.fold == str(fold)) \
                & (pd.to_datetime(logs.time) > since)
                ]\
                .sort_values('task_f1', ascending = False )\
                .head(n)

    df_raw = df.copy(deep = True)
    df = df.select_dtypes(include=np.number).astype(float)
    df['fold'] = df_raw.fold.apply(lambda x : np.float64(x) if x != 'all' else 99)
    df['time'] = df_raw.time
    return df

# cols = [
#         'task_f1',
#         'dropout', 
#          'dropout',
#          'fold',
#         'epochs',
#         'n_layers',
#         'neg_weight',
#         'pos_weight',
#         'pad_weight',
#         'batch_size'
#         ]

def plot_hparams(df, cols, scaled = True):

    scaler = MinMaxScaler()

    for col in cols:
        try:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            print('added ', col)
        except:
            df.drop(columns=[col], inplace = True)
            print('dropped ', col)


    df.reset_index(inplace = True)

    # ax = plt.figure()
    fig, ax = plt.subplots(figsize=(20,10))

    pd.plotting.parallel_coordinates(

        df[cols].reset_index(), 'index' , ax = ax, color = sns.cubehelix_palette(df[cols].shape[0], reverse = True)

    )

    plt.show()


now = datetime.now()
final_design_optimisation = pd.to_datetime('2021-01-25 12:25:47.657977')


def make_params(since):

    mdf = pd.DataFrame()

    for method in ['ALBERT', 'ELECTRA', 'ROBERTA']:
        bdf = pd.concat([get_best(method, n, since = since, n=1) for n in list(range(5)) + ['all']], axis = 0)
        bdf['method'] = method
        mdf = pd.concat([mdf,bdf])


    mdf.fold = mdf.fold.apply(lambda x : str(int(x)))
    mdf = mdf.set_index(['method','fold'])

    mdf[['task_f1','epochs']]
    #%%

    mdf[['batch_size', 'epochs', 'n_layers']] = mdf[['batch_size', 'epochs', 'n_layers']].astype(int)

    return mdf

run_26_01_21 = pd.to_datetime('2021-01-26 12:20:24.727613')
BEST = make_params(run_26_01_21).to_dict(orient='index')
# %%
best = make_params(final_design_optimisation)
current = make_params(run_26_01_21)
# %%
