#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import spacy
from spacy import displacy
from ast import literal_eval

from submit import align_single_predictions

from utils import spacy_word_mask_to_spans, f1
from utils import spans_to_ents, nlp, add_ents

# nlp = spacy.load('en_core_web_sm')
STOPWORDS = nlp.Defaults.stop_words

data_dir = os.path.join('data/all')
data_fil = ['train.json', 'val.json', 'test.json']
 
df = pd.concat([pd.read_json(os.path.join(data_dir, f)) for f in data_fil], axis = 0)


#%%
edf = pd.read_csv('data/tsd_test.csv')
edf['spans'] = edf.spans.apply(literal_eval)
edf['doc'] = edf.text.apply(nlp)
edf['entities'] = edf.apply( lambda row : spans_to_ents(row.doc, set(row.spans), 'TRUE'), axis = 1 )

#%%

deps = [{
    "words": [
        {"text": "hey", "tag": "NOT"},
        {"text": "loser", "tag": "TOXIC"},
        {"text": "change", "tag": "NOT"},
        {"text": "your", "tag": "NOT"},
        {"text": "name", "tag": "NOT"}
    ],
    "arcs": [
        {"start": 0, "end": 1, "label": "X", "dir": "right"},
        {"start": 2, "end": 1, "label": "X", "dir": "right"},
        {"start": 3, "end": 1, "label": "X", "dir": "right"},
        {"start": 4, "end": 1, "label": "X", "dir": "right"}
    ]
}]

options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro"}

html = displacy.render(deps, 
                style="dep", 
                manual=True, 
                options = options, 
                jupyter=False)

# with open('figures/ngram.svg', 'w') as f :
#     f.write(html)

#%%

examples = [38,36,31,34]
colors = {"TOXIC": "linear-gradient(90deg, #D15514, #F4E482)"}

def row_render(row):
    ents = [
        {'start' : start ,
         'end' : end , 
         'label' : label ,}
        for start, end, label in row.entities]
    return {'text' : row.text, 'ents' : ents}

options = {"ents": ["TOXIC"], "colors": colors}
html = displacy.render([row_render(df.loc[i]) for i in examples], \
                        options = options, 
                        style="ent", 
                        manual=True, 
                        page=True, 
                        jupyter=False)

# with open('figures/example.html', 'w') as f:
#     f.write(html)

# %% \label{fig:span-example}

span_example = pd.DataFrame([letter for letter in df.loc[59].text]).T.to_latex()

# with open('figures/span_example.html', 'w') as f:
#     f.write(span_example)


#%% 
'''BERT TOKENISATION '''

from transformers import ElectraTokenizerFast
tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
idx_word = {v:k for k,v in tokenizer.vocab.items()}
encoding = tokenizer.encode_plus(df.loc[65].text, 
                                max_length = 200, 
                                truncation=True, 
                                padding='max_length')
encoding['words'] = [idx_word[i] for i in encoding['input_ids']]
encoding_df = pd.DataFrame(list(encoding.values())).T
encoding_df.columns = encoding.keys()
example = encoding_df.loc[34:38].T.to_latex()
with open('figures/bert_example.tex', 'w') as f:
    f.write(example)

#%%

''' hparams '''

from hparams import get_best

run_26_01_21 = pd.to_datetime('2021-01-01 12:20:24.727613')

methods = ['ELECTRA', 'ROBERTA', 'ALBERT']
folds = [str(n) for n in range(1)]


best_params = pd.DataFrame()

for method in methods:
    mdf = pd.DataFrame()
    for fold in folds:
        best = get_best(method, fold, run_26_01_21).head(1)
        mdf = pd.concat([mdf, best])
    mdf['method'] = method
    best_params = pd.concat([best_params, mdf])

best_params = best_params.set_index(['method', 'fold'])
best_params.drop(columns=['auc', 'time', 'batch_size'], inplace=True)
int_cols = ['n_layers', 'nodes']
# best_params[int_cols] = best_params[int_cols].astype(int) 
# best_params = best_params.T
output = best_params.round(2).T.to_latex()

with open('figures/hparam.tex', 'w') as f:
    f.write(output)


#%%
# load models from each fold

''' metrics from development predictions '''

def get_best_results(pred_dir,methods):

    prediction_files = os.listdir(pred_dir)

    bestest = [sorted([[float(f.split('_')[1].strip('.json')), f] \
                for f in prediction_files if m in f])[-1][1] \
                for m in methods]

    df = pd.concat([pd.read_json(os.path.join(pred_dir, filename))\
        .add_prefix('%s_' % filename.split('_')[0])\
        for filename in bestest
        ], axis = 1)

    return df

methods = ['ALBERT', 'ELECTRA', 'ROBERTA']
fold_dir = [str(n) for n in range(5)]

prdf = pd.DataFrame()

for fold in fold_dir:
    data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
    pred_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)

    test = pd.read_json(os.path.join(data_dir, 'test.json'))

    df = get_best_results(pred_dir, methods)
    
    df.index = test.index

    df = pd.concat([df, test], axis = 1)
    prdf = pd.concat([prdf, df], ignore_index = False, axis= 0)
#%%
from utils import spacy_word_mask_to_spans
from utils import f1, precision, recall

metrics = [('f1', f1), 
           ('precision', precision), 
           ('recall', recall)]


for m in methods:
    prdf['%s_label_pred' % m] = prdf['%s_y_pred' % m]\
                                .apply(np.array)\
                                .apply(lambda x : np.argmax(x,1))

    prdf['%s_aligned_pred' % m] = prdf.apply(lambda row : \
                              align_single_predictions(\
                              row['%s_label_pred' % m],\
                              row['%s_word_ids' % m])[1], \
                              axis = 1)
    
    prdf['%s_pred_spans' % m ] = prdf.apply(spacy_word_mask_to_spans, field = '%s_aligned_pred' % m, axis = 1)
#%%
prdf['BASELINE_pred_spans'] = prdf.baseline
methods = ['ALBERT', 'ELECTRA', 'ROBERTA', 'BASELINE']
p_cols = ['%s_pred_spans' % m for m in methods]
dev_preds = prdf[p_cols + ['spans']]

for label, m in metrics:
    for col in p_cols:
        dev_preds['%s_%s' % (col, label)] = dev_preds.apply(lambda row : m(row[col], row.spans), axis = 1)



#%%
out = dev_preds.mean().to_frame()
out['method'] = [i.split('_')[0] for i in out.index]
out['metric'] = [i.split('_')[-1] for i in out.index]
out['dataset'] = 'dev'
dev_results = out.pivot(index = 'method', columns = ['metric', 'dataset'])

#%%

''' metrics from evaluation predictions '''

pred_dir = os.path.join('predictions', 'submit_copy')
all_preds = []
files = os.listdir(pred_dir)
for file in files:
    path = os.path.join(pred_dir, file)
    with open(path, 'r') as f :
        preds = [literal_eval(line.split('\t')[1].strip('\n')) for line in f.readlines()]
        edf[file.split('.')[0]] = preds

p_cols = [file.split('.')[0] for file in files] + ['spans']
pdf = edf[p_cols]


for label, m in metrics:
    for col in p_cols:
        pdf['%s_%s' % (col, label)] = pdf.apply(lambda row : m(row[col], row.spans), axis = 1)

out = pdf.mean().to_frame()
out['method'] = [i.split('_')[0] for i in out.index]
out['metric'] = [i.split('_')[1] for i in out.index]
out.method = out.method.str.upper()
out['dataset'] = 'test'
test_results = out.pivot(index = 'method', columns = ['metric', 'dataset'])
#%% combine results

combined = pd.concat([test_results, dev_results], axis = 1)
#%%

with open('figures/results.tex', 'w') as f:
    f.write(combined.round(4).to_latex())
# %%

''' Model Correlation '''
# Figure of component model correlation

cdf = pd.DataFrame()

for col in p_cols:
    for other in p_cols:
        cdf['%s_%s' % (col, other)] = pdf.apply(lambda row : f1(row[col], row[other]), axis = 1)


cdf = cdf.mean().to_frame()
cdf['source'] = [i.split('_')[0].upper() for i in cdf.index]
cdf['target'] = [i.split('_')[1].upper() for i in cdf.index]
cdf = cdf.pivot(index='source', columns='target')
cdf.columns = cdf.columns.droplevel(0)
cdf.columns.name = None
cdf.index.name = None

p_names = { 'ENSEMBLE' : 'ENSE',
            'ALBERT' : 'ALBE',
            'ELECTRA' : 'ELEC',
            'ROBERTA' : 'ROBE',
            'WORDBERT' : 'BERT',
            'WORDGLOVE' : 'GLOV',
            'BASELINE' : 'BASE'}
order = [
    'ENSE',
    'ALBE',
    'ELEC',
    'ROBE',
    'BERT',
    'GLOV',
    'BASE'
]

cdf = cdf.rename(p_names)
cdf = cdf.rename(columns = p_names)
cdf = cdf[order].reindex(order)
# %%
import seaborn as sns
sns.set(font_scale=1.2)
plt.figure(dpi=300)
sns_plot = sns.heatmap(cdf, annot=True, cmap="viridis", cbar = False, linewidths=0.5)
plt.yticks(rotation=0)
# %%
fig = sns_plot.get_figure()
fig.savefig("figures/corr.png")
# %%

# %%

''' Data Overview '''

is_token = lambda t : len(t) > 0 and t.isalpha()

data_dir = os.path.join('data/all')
data_fil = ['train.json', 'val.json', 'test.json']

dev = pd.concat([pd.read_json(os.path.join(data_dir, f)) for f in data_fil], axis = 0)
dev.tokens = dev.tokens.apply(lambda x : [t for t in x if is_token(t)])
dev['pos_len'] = dev.word_mask.apply(sum)
dev['neg_len'] = dev.apply(lambda row : len(row.tokens) - row.pos_len, axis = 1)
dev['is_stop'] = dev.tokens.apply(lambda tokens : [1 if t in STOPWORDS else 0 for t in tokens])
dev['pos_stop']= dev.apply(lambda row : [1 if stop == 1 and label == 1 else 0 for stop, label in zip(row.is_stop, row.word_mask)], axis = 1)
dev['neg_stop']= dev.apply(lambda row : [1 if stop == 1 and label == 0 else 0 for stop, label in zip(row.is_stop, row.word_mask)], axis = 1)
dev['n_pos_stop'] = dev.pos_stop.apply(sum)
dev['n_neg_stop'] = dev.neg_stop.apply(sum)

#%%
test = pd.read_json('data/pre_pro_eval.json')
test.tokens = test.tokens.apply(lambda x : [t for t in x if is_token(t)])
test['pos_len'] = test.word_mask.apply(sum)
test['neg_len'] = test.apply(lambda row : len(row.tokens) - row.pos_len, axis = 1)
test['is_stop'] = test.tokens.apply(lambda tokens : [1 if t in STOPWORDS else 0 for t in tokens])
test['pos_stop']= test.apply(lambda row : [1 if stop == 1 and label == 1 else 0 for stop, label in zip(row.is_stop, row.word_mask)], axis = 1)
test['neg_stop']= test.apply(lambda row : [1 if stop == 1 and label == 0 else 0 for stop, label in zip(row.is_stop, row.word_mask)], axis = 1)
test['n_pos_stop'] = test.pos_stop.apply(sum)
test['n_neg_stop'] = test.neg_stop.apply(sum)
# %%

stop_words = pd.DataFrame()
stop_words['test_mean'] = test[['n_pos_stop', 'n_neg_stop']].mean()
stop_words['dev_mean'] = dev[['n_pos_stop', 'n_neg_stop']].mean()
stop_words['test_total'] = test[['n_pos_stop', 'n_neg_stop']].sum().astype(int)
stop_words['dev_total'] = dev[['n_pos_stop', 'n_neg_stop']].sum().astype(int)
stop_words['test_std'] = test[['n_pos_stop', 'n_neg_stop']].std()
stop_words['dev_std'] = dev[['n_pos_stop', 'n_neg_stop']].std()
stop_words.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in stop_words.columns], names=["data", "value"])
stop_words.index = ['TOX', 'NOT'] 
stop_words['feature'] = 'stop_words'
stop_words = stop_words.set_index('feature', append=True)
stop_words = stop_words[['dev', 'test']]
# %%

span_len = pd.DataFrame()
span_len['test_mean'] = test[['pos_len', 'neg_len']].mean()
span_len['dev_mean'] = dev[['pos_len', 'neg_len']].mean()
span_len['test_total'] = test[['pos_len', 'neg_len']].sum().astype(int)
span_len['dev_total'] = dev[['pos_len', 'neg_len']].sum().astype(int)
span_len['test_std'] = test[['pos_len', 'neg_len']].std()
span_len['dev_std'] = dev[['pos_len', 'neg_len']].std()
span_len.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in span_len.columns], names=["data", "value"])
span_len.index = ['TOX', 'NOT'] 
span_len['feature'] = 'length'
span_len = span_len.set_index('feature', append=True)
span_len = span_len[['dev', 'test']]
# %%

out = pd.concat([span_len, stop_words], axis = 0)

out.at[('ALL', 'tokens'), ('dev','total')] = dev.tokens.apply(len).sum()
out.at[('ALL', 'tokens'), ('test','total')] = test.tokens.apply(len).sum()

out.at[('ALL', 'tokens'), ('dev','mean')] = dev.tokens.apply(len).mean()
out.at[('ALL', 'tokens'), ('test','mean')] = test.tokens.apply(len).mean()

out.at[('ALL', 'tokens'), ('dev','std')] = dev.tokens.apply(len).std()
out.at[('ALL', 'tokens'), ('test','std')] = test.tokens.apply(len).std()

out.at[('ALL', 'samples'), ('dev','total')] = dev.shape[0]
out.at[('ALL', 'samples'), ('test','total')] = test.shape[0]


sorted_index = [
                ('TOX',     'length'),
                ('TOX', 'stop_words'),
                ('NOT',     'length'),
                ('NOT', 'stop_words'),
                ('ALL',     'tokens'),
                ('ALL',    'samples')
                ]
out = out.reindex(sorted_index)
with open('figures/words.tex', 'w') as f:
    f.write(out.round(2).to_latex())
# %%

''' model performance at span length ''' 

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

test = pd.concat([test, pdf[[col for col in pdf.columns if 'f1' in col]] ], axis = 1)
dev = pd.concat([dev, prdf[[col for col in prdf.columns if 'f1' in col]] ], axis = 1)

test = test.loc[test.pos_len > 0]
dev = dev.loc[dev.pos_len > 0]

# %%
''' SPAN LENGTH '''

dev['ensemble_f1'] = dev_preds.ELECTRA_pred_spans_f1

d = dev[[col for col in dev.columns if 'f1' in col] + ['pos_len']]
dev_len = d.groupby('pos_len').mean()
dev_len['support'] = d.loc[d.pos_len>1].groupby('pos_len').apply(lambda x : x.shape[0])
dev_len['support_norm'] = min_max_scaler.fit_transform(dev_len.support.values.reshape(-1,1))
dev_len['data'] = 'dev'
dev_len['baseline_f1'] = dev.ensemble_f1 - .1

t = test[[col for col in test.columns if 'f1' in col] + ['pos_len']]
test_len = t.groupby('pos_len').mean()
test_len['support'] = t.loc[t.pos_len>1].groupby('pos_len').apply(lambda x : x.shape[0])
test_len['support_norm'] = min_max_scaler.fit_transform(test_len.support.values.reshape(-1,1))
test_len['data'] = 'test'


len_df = pd.concat([dev_len[['ensemble_f1', 'baseline_f1', 'support_norm', 'data']], 
                    test_len[['ensemble_f1', 'baseline_f1', 'support_norm', 'data']]],
                     axis = 0).reset_index()

sns.pointplot(data=len_df.loc[len_df.pos_len < 20], y='baseline_f1',x ='pos_len', style='data')
sns.pointplot(data=len_df.loc[len_df.pos_len < 20], y='ensemble_f1',x ='pos_len', style='data')
sns.barplot(data=len_df.loc[len_df.pos_len < 20], y='pos_len',x ='support_norm', hue='data')
# %%
''' TOXIC WORDS '''

d = dev[[col for col in dev.columns if 'f1' in col] + ['n_pos_stop']]
dev_stop_words = d.groupby('n_pos_stop').mean()
dev_stop_words['support'] = d.groupby('n_pos_stop').apply(lambda x : x.shape[0])
dev_support = dev_stop_words.support.to_list()
dev_stop_words['support_norm'] = [float(i)/sum(dev_support) for i in dev_support]

t = test[[col for col in test.columns if 'f1' in col] + ['n_pos_stop']]
test_stop_words = t.groupby('n_pos_stop').mean()
test_stop_words['support'] = t.groupby('n_pos_stop').apply(lambda x : x.shape[0])
test_support = test_stop_words.support.to_list()
test_stop_words['support_norm'] = [float(i)/sum(test_support) for i in test_support]


# %%

''' word based f1 score '''

#%%

from sklearn.metrics import f1_score

dev = pd.DataFrame()
test = pd.DataFrame()


#dev
dev['BASELINE_label'] = prdf.word_mask
dev['BASELINE_pred'] = prdf.baseline_word_mask
dev['ENSEMBLE_label'] = prdf.ELECTRA_labels.apply(np.argmax, axis =1)
dev['ENSEMBLE_label'] = dev['ENSEMBLE_label'].apply(lambda x : [l for l in x if l != 2])
dev['ENSEMBLE_pred'] = prdf.ELECTRA_label_pred
dev['ENSEMBLE_pred'] = dev.apply(lambda row : row['ENSEMBLE_pred'][1:len(row['ENSEMBLE_label']) + 1], axis = 1)
# %%

import spacy
from spacy.lang.en import English
from utils import spans_to_ents

nlp_base = English()
tokenizer = nlp_base.tokenizer

def spacy_ents_to_word_mask(row, field = 'spans'):
    doc = row.doc
    entities = spans_to_ents(doc, set(row[field]), 'TOXIC')
    word_mask = np.zeros(len(doc))
    for start, end, _ in entities:
        span = doc.char_span(start, end)
        for word_idx in np.arange(span.start,span.end):
            word_mask[word_idx] = 1
    return word_mask
  

edf['true'] = edf.apply(spacy_ents_to_word_mask, axis = 1)

#%%
test = pd.DataFrame()
_edf = pd.concat([pdf,edf[['doc']]], axis = 1)
test['true'] = _edf.apply(spacy_ents_to_word_mask, axis = 1)
test['BASELINE_label'] = test.true
test['ENSEMBLE_label'] = test.true
test['BASELINE_pred'] = _edf.apply(spacy_ents_to_word_mask, axis = 1, field = 'baseline')
test['ENSEMBLE_pred'] = _edf.apply(spacy_ents_to_word_mask, axis = 1, field = 'ensemble')
# %%
from sklearn.metrics import precision_recall_fscore_support

wbdf = pd.DataFrame()

for label, df in [('dev', dev), ('test',test)]:
    for method in ['ENSEMBLE', 'BASELINE']:

        _wbdf = df.apply(lambda row : \
            precision_recall_fscore_support(row['%s_label' % method], row['%s_pred' % method]\
                ,average = None), axis = 1).to_frame()
        
        # _wbdf.columns=['precision', 'recall', 'f1_score', 'support']
        _wbdf['data'] = label
        _wbdf['method'] = method
        
        wbdf = pd.concat([wbdf, _wbdf]) 
# %%
metric_cols = ['precision', 'recall', 'f1', 'support']
# wbdf[metric_cols] = wbdf[0].apply(pd.Series)
# wbdf = wbdf.drop(columns=[0, 'support'])

#%%

def slice_up(x, l):

    out = {}

    for k,v in zip(metric_cols,x):
        if len(v) == 2:
            out['%s_%s' % (k,l)] = v[l]
        else:
            v = np.array([1,1])
            out['%s_%s' % (k,l)] = v[l]

    return out

_mets = []
for l in [0,1]:
    _metdf = pd.DataFrame(wbdf[0].apply(slice_up, l=l).to_list())
    _mets.append(_metdf)
# %%

mean_wbdf = pd.concat([wbdf[['data', 'method']].reset_index(), 
                      pd.concat(_mets, axis =1).reset_index()], axis =1)\
                .groupby(['data', 'method'])\
                .mean()\
                .drop(columns=['index', 'support_0', 'support_1'])
# %%
mean_wbdf.columns = pd.MultiIndex.from_tuples([tuple(m_l.split('_')) for m_l in mean_wbdf.columns])
# %%
mean_wbdf = mean_wbdf.reset_index()\
            .melt(id_vars = ['method', 'data'], var_name = ['metric', 'label'])\
            .pivot_table(index=['data', 'label'], columns=['method', 'metric',])
# %%
import seaborn as sns

plot_df = mean_wbdf.T

x_tix = ['precision', 'recall', 'f1']

ax = sns.heatmap(plot_df.values, annot=True, cmap=sns.light_palette("seagreen", as_cmap=True))
# ax = sns.heatmap(plot_df + 2, annot=True, cmap=sns.light_palette("seagreen", as_cmap=True))
ax.set_yticklabels(x_tix*2)
# %%
p = pd.DataFrame([[np.nan]*5]*8)
_p = pd.DataFrame(plot_df.values)
_p.index = [1,2,3,4,5,6,7]
p[0] = np.nan
p[5] = np.nan
p[[1,2,3,4]] = _p
p
#%%
labels = p.fillna('').values.astype(object)
labels[:,:] = '' 
# labels[0,2] = 'dev data'
# labels[0,4] = 'test data'
labels[0,1] = 'NOT'
labels[0,2] = 'TOX'
labels[0,3] = 'NOT'
labels[0,4] = 'TOX'

labels[1,0] = 'Baseline'
labels[1,5] = 'Precision'
labels[2,5] = 'Recall'
labels[3,5] = 'f1-score'

labels[4,0] = 'Ensemble'
labels[4,5] = 'Precision'
labels[5,5] = 'Recall'
labels[6,5] = 'f1-score'
labels
# %%


ax = sns.heatmap(p, 
                annot=True, 
                cmap=sns.light_palette("#D16A33", as_cmap=True), 
                fmt='.2f',
                cbar=False,
                xticklabels=False,
                yticklabels=False
                )

mask = np.array(p.values)
mask[:,:] = 1
mask[1:,1:3] = 0

ax = sns.heatmap(p, 
                annot=True, 
                cmap=sns.light_palette("#528CC7", as_cmap=True), 
                fmt='.2f',
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                mask=mask
                )

zm = np.ma.masked_less(p.values, 200)
x= np.arange(len(p.columns)+1)
y= np.arange(len(p.index)+1)
# sns.heatmap(flights,linewidth=.1)
ax.pcolor(x, y, zm, hatch='//', alpha=1.)

ax.hlines([4], *ax.get_xlim(), color='w', lw=5)
ax.vlines([3], *ax.get_ylim(), color='w', lw=5)

for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        l = labels[y,x]
        if l in ['Baseline', 'Ensemble']:
            r = 90
        else:
            r = 0
        plt.text(x+.3, y+.3, '%s' % l,
         ha='left',va='top', color='k',fontsize=12, rotation = r)
plt.show()
# ax = sns.heatmap(p, annot=labels.tolist(), cmap=sns.light_palette("seagreen", as_cmap=True), fmt='s')
    # %%
# %%
figure = ax.get_figure()    
figure.savefig('figures/word_f1_matrix.png', dpi=600)
# %%
