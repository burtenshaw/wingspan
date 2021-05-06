#%%
import pandas as pd
import spacy
from spacy.tokens import Span
import os
import numpy as np
os.chdir('/home/burtenshaw/now/spans_toxic')

nlp = spacy.load("en_core_web_sm")

data = pd.read_pickle('data/train.bin')

# %%


#%%

data = data[:20]
data['entities'] = data.apply( lambda row : spans_to_ents(nlp(row.text), set(row.spans), 'TOXIC'), axis = 1 )
data['doc'] = data.text.apply(nlp)
data['doc_ents'] = data.apply(lambda row : [row.doc.char_span(start, end, label = label) for start, end, label in row.entities ], axis =1)
data['doc'] = data.apply(add_ents, axis = 1)
data['word_mask'] = data.doc.apply(spacy_ents_to_word_mask)
data['parsing_predictions'] = data.apply(spacy_word_mask_to_spans, axis = 1)
# %%

