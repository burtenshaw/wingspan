#%%

import pandas as pd
import json_lines




#%%

import pandas as pd
import numpy as np
import os
import json

from utils import spacy_word_mask_to_spans, f1
from utils import spans_to_ents, nlp

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
