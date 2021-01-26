#%%
import pandas as pd
import numpy as np
from utils import *
#%%
predictions_list = []

for fold in range(5):
    fold = str(fold)
    spacy = pd.read_json('/home/burtenshaw/now/spans_toxic/predictions/%s/spacy_sentiment.json' % fold)
    spacys = []
    for s in spacy.fillna(-1).values.tolist():
        _s = []

        for p in s:
            if p >= 0:    
                _s.append(p)

        spacys.append(_s)

    test = pd.read_pickle('/home/burtenshaw/now/spans_toxic/data/%s/test.bin' % fold)

    test['spacy'] = spacys

    predictions_list.append(test.spacy)

# %%
df = pd.concat(predictions_list)
df.to_pickle('/home/burtenshaw/now/spans_toxic/data/all/spacy.bin')
# %%
data = df
data['doc'] = data.text.apply(nlp)

data['tokens'] = data.doc.apply(lambda doc : [token.text for token in doc])
data.drop(columns = ['doc_ents', 'doc'], inplace = True)
# %%
