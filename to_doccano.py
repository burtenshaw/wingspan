#%%

import pandas as pd
import json_lines
import numpy as np
import os
import json
from ast import literal_eval
from utils import spacy_word_mask_to_spans, f1
from utils import spans_to_ents, nlp, add_ents
import ast
import csv
import itertools
import string
import sys

SPECIAL_CHARACTERS = string.whitespace

def _contiguous_ranges(span_list):
    """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)]."""
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output


def fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS):
    """Applies minor edits to trim spans and remove singletons."""
    cleaned = []
    for begin, end in _contiguous_ranges(spans):
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned

def to_doccano(entities, text):
    ''' 
    {
        "text": "EU rejects German call to boycott British lamb.", 
        "labels": [ [0, 2, "ORG"], [11, 17, "MISC"], ... ]
        }  
        
        '''
    return { "text" : text, "labels": list(entities)}

def write_doccano_dataset(pr_df, fold = 'all'):
    
    pr_df['TRUE_pred_spans'] = pr_df.spans

    for method in ['ELECTRA', 'ROBERTA', 'BASELINE', 'ALBERT', 'TRUE']:
        pr_df['doc'] = pr_df.text.apply(nlp)
        pr_df['%s_entities' % method] = pr_df.apply( lambda row : \
            spans_to_ents(row.doc, set(row['%s_pred_spans' % method]), method), axis = 1 )

    output_path = os.path.join('/home/burtenshaw/now/spans_toxic/data/doccano')

    singles = []

    for method in ["ELECTRA", "ROBERTA", "BASELINE", 'ALBERT', 'TRUE']:
        col = '%s_entities' % method
        output = pr_df[[col, 'text']].apply(lambda row : (row.text, row[col]), axis = 1).to_list()
        singles.append(output)

    together = []

    for z in zip(*singles):
        labels = [[list(e) for e in m[1]] for m in z]
        labels = [item for sublist in labels for item in sublist]
        text = z[0][0]
        together.append(to_doccano(labels, text))

    with open(os.path.join(output_path, 'models_%s.jsonl' % fold), 'w') as f:
        for line in together[:2500]:
            f.write('%s\n' % json.dumps(line))

# %%
# df = pd.read_pickle('data/all/eval.bin')
# sub_path = '/home/burtenshaw/now/spans_toxic/predictions/submit/spans-pred_ELECTRA_0.6578636624699009.txt'
# with open(sub_path, 'r') as f :
#     spans = f.readlines()
# df['spans'] = spans
# df.spans = df.spans.apply(lambda x : literal_eval(x.split('\t')[1]))
# df['clean_spans'] = df.apply(lambda row : fix_spans(row.spans, row.text), axis = 1)

df = pd.read_csv('data/tsd_test.csv')
df['spans'] = df.spans.apply(literal_eval)

#%%
df['doc'] = df.text.apply(nlp)
df['entities'] = df.apply( lambda row : spans_to_ents(row.doc, set(row.spans), 'TRUE'), axis = 1 )
# %%
output = df.apply(lambda row : to_doccano(row.entities, row.text), axis = 1).to_list()
output_path = os.path.join('/home/burtenshaw/now/spans_toxic/data/doccano')

with open(os.path.join(output_path, 'true_labels.jsonl'), 'w') as f:
    for line in output:
        f.write('%s\n' % json.dumps(line))
# %%
data= []
with open(os.path.join(output_path, 'electra_check.json'), 'r') as f:
    for line in f.readlines():
        data.append(json.loads(line.strip('\n'))['labels'])
# %%

df['checked'] = data
# %%
def ents_to_spans(ents, text, special_characters=SPECIAL_CHARACTERS):
    """Applies minor edits to trim spans and remove singletons."""
    cleaned = []
    for begin, end, _ in ents:
        end -= 1
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned

df['checked_cleaned'] = df.apply(lambda row : ents_to_spans(row.checked, row.text), axis = 1)
# %%
df['cont_ranges'] = df.spans.apply(_contiguous_ranges)
out = df.checked_cleaned.to_list()
# %%
from submit import to_submit

to_submit(out, output_path='/home/burtenshaw/now/spans_toxic/predictions/submit/spans-pred_electra_cleaned.txt')
# %%
