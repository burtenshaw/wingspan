#%%
import pandas as pd
import json
import json_lines
import os 
from ast import literal_eval
import spacy
os.chdir('/home/burtenshaw/now/spans_toxic')
from utils import spans_to_ents
nlp = spacy.load("en_core_web_sm")
from toxic_spans.evaluation.fix_spans import fix_spans

text_path = os.path.join('/home/burtenshaw/now/spans_toxic/data', 'tsd_test.csv')
model_path = os.path.join('/home/burtenshaw/now/spans_toxic/data', 'baseline_2.csv')
output_path = os.path.join('/home/burtenshaw/now/spans_toxic/data', 'test_doccano.jsonl')
#%%
test = pd.DataFrame()
test['text'] = pd.read_csv(text_path).text
test['spans'] = pd.read_csv(model_path, index_col = 0)
test.spans = test.spans.apply(literal_eval)
#%%
test['doc'] = test.text.apply(nlp)
test['entities'] = test.apply( lambda row : spans_to_ents(row.doc, set(row.spans), 'TOXIC'), axis = 1 )
output = test[['text','entities']].rename(columns={'entities':'labels'}).to_dict(orient = 'records')
    
with open(output_path, 'w') as f:
    for line in output:
        # line['labels'] = [[s,e,'MISC'] for s,e,_ in line['labels']]
        f.write(json.dumps(line) + '\n')


spans = pd.read_csv(model_path, index_col=0)
span_list = spans['0'].to_list()

# %%
def to_submit(span_list):
    output = ''
    for n, line in enumerate(span_list):
        output += '%s\t%s\n' % (n,line)

    with open('predictions/submit/spans-pred.txt', 'w') as f:
        f.write(output)


def from_doccano():
    anno = []
    with open('/home/burtenshaw/now/spans_toxic/data/doccano/file.json1', 'rb') as f:
        for item in json_lines.reader(f):
            anno.append(item)

    anno = pd.DataFrame(anno)
    anno['_text'] = pd.read_csv(text_path).text

    return anno

def ents_to_spans(ents):
    pred_spans = []
    for start_char, end_char, _ in ents:
        pred_spans.extend(range(start_char, end_char))
    return pred_spans


anno = from_doccano()
anno.labels = anno.labels.apply(ents_to_spans)
anno['labels'] = anno.apply(lambda row : fix_spans(row.labels, row.text), axis = 1)
to_submit(anno.labels.to_list())

# # # %%
# # anno.loc[_anno_labels != anno.labels]
# # # %%

# # %%
# from utils import spacy_word_mask_to_spans
# anno['doc'] = anno.text.apply(nlp)
# anno['tokens'] = anno.doc.apply(lambda doc : [token.text for token in doc])
# anno['labels'] = anno.doc.apply(spacy_ents_to_word_mask)
# # %%
# # anno.drop(columns=['doc'], inplace=True)
# anno[['tokens','labels']].to_json('data/anno_13_1.json')
# # %%
#%%
to_submit(test.spans.to_list())