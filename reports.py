#%% 
import pandas as pd
import numpy as np
import os
import json

from sklearn.metrics import classification_report
from submit import align_single_predictions
from utils import spacy_word_mask_to_spans, f1
from utils import spans_to_ents, nlp
import to_doccano
#%%

fold_dir = ['all']
fold_dir = [str(n) for n in range(5)]

_reports = []
_predictions = []
_word_level = []

for fold in fold_dir:
    data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
    pred_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)

    test = pd.read_pickle(os.path.join(data_dir, 'test.bin'))

    df = pd.concat([
        pd.read_json(os.path.join(pred_dir, 'ELECTRA.json')).add_prefix('electra_'),
        pd.read_json(os.path.join(pred_dir, 'ROBERTA.json')).add_prefix('roberta_'),
        pd.read_json(os.path.join(pred_dir, 'ALBERT.json')).add_prefix('albert_'),
        test.baseline_word_mask,
        test.text,
        test.spans
        ], axis = 1)
    #%%

    df['electra_y_pred'] = df.electra_y_pred.apply(np.array).apply(lambda x : np.argmax(x,1))
    df['electra_aligned_pred'] = df.apply(lambda row : align_single_predictions(row.electra_y_pred, row.electra_word_ids)[1], axis = 1)
    df['roberta_y_pred'] = df.roberta_y_pred.apply(np.array).apply(lambda x : np.argmax(x,1))
    df['roberta_aligned_pred'] = df.apply(lambda row : align_single_predictions(row.roberta_y_pred, row.roberta_word_ids)[1], axis = 1)
    df['albert_y_pred'] = df.albert_y_pred.apply(np.array).apply(lambda x : np.argmax(x,1))
    df['albert_aligned_pred'] = df.apply(lambda row : align_single_predictions(row.albert_y_pred, row.albert_word_ids)[1], axis = 1)
    
    df['baseline_aligned_pred'] = df.baseline_word_mask

    #%%
    wdf = pd.DataFrame()
    wdf['ELECTRA'] = df.electra_aligned_pred.apply(lambda x : x[:128]).explode()
    wdf['ROBERTA'] = df.roberta_aligned_pred.apply(lambda x : x[:128]).explode()
    wdf['ALBERT'] = df.albert_aligned_pred.apply(lambda x : x[:128]).explode()

    wdf['BASELINE'] = test.baseline_word_mask.apply(np.array)\
                              .apply(lambda x : np.pad(x[:128],(0,128 - x[:128].shape[0])))\
                              .explode().values

    wdf['y'] =  test.word_mask.apply(np.array)\
                            .apply(lambda x : np.pad(x[:128],(0,128 - x[:128].shape[0])))\
                            .explode().values
    _word_level.append(wdf)
    wdf = wdf.astype(int)

    for method in ['ELECTRA', 'ROBERTA', 'BASELINE', 'ALBERT']:
        cheap = pd.DataFrame(classification_report(wdf.y.values, wdf[method].values, output_dict=True))
        cheap['fold'] = fold
        cheap['method'] = method
        df['%s_pred_spans' % method] = df.apply(spacy_word_mask_to_spans, \
                                    field = '%s_aligned_pred' % method.lower(), axis = 1)

        cheap['task_f1'] = df.apply(lambda row : f1(row['%s_pred_spans' % method], row.spans) , axis = 1).mean()
        
        _reports.append(cheap)
        _predictions.append(df)

    
#%%
reports = pd.concat(_reports, axis = 0)
wdf = pd.concat(_word_level, axis = 0)

# %%
for n, df in enumerate(_predictions):
    to_doccano.write_doccano_dataset(df, fold=n)
# %%
wdf.corr()
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
X = np.vstack(wdf.drop(columns=['y']).values)
y = wdf.y.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_leaf = [1, 2, 4]
min_samples_split = [2, 5, 10]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(X_train, y_train)


#%%

y_pred = clf.predict(X_test)

pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
# %%
