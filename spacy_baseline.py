#%%
import ast
import csv
import random
import statistics
import sys
import os
import pandas as pd
import sklearn
import spacy

from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    # ipython convenience
    mgc = get_ipython().magic
    mgc(u'%load_ext autoreload')
    mgc(u'%autoreload 2')
    os.chdir('/home/burtenshaw/now/spans_toxic')
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append('toxic_spans/evaluation')

import semeval2021
import fix_spans

#%%
def spans_to_ents(doc, spans, label):
  """Converts span indicies into spacy entity labels."""
  started = False
  left, right, ents = 0, 0, []
  for x in doc:
    if x.pos_ == 'SPACE':
      continue
    if spans.intersection(set(range(x.idx, x.idx + len(x.text)))):
      if not started:
        left, started = x.idx, True
      right = x.idx + len(x.text)
    elif started:
      ents.append((left, right, label))
      started = False
  if started:
    ents.append((left, right, label))
  return ents


def main(train, test):
  """Train and eval a spacy named entity tagger for toxic spans."""

  # Convert training data to Spacy Entities
  nlp = spacy.load("en_core_web_sm")

  print('preparing training data')
  training_data = []
  for n, (spans, text) in enumerate(train):
    doc = nlp(text)
    ents = spans_to_ents(doc, set(spans), 'TOXIC')
    training_data.append((doc.text, {'entities': ents}))

  toxic_tagging = spacy.blank('en')
  toxic_tagging.vocab.strings.add('TOXIC')
  ner = nlp.create_pipe("ner")
  toxic_tagging.add_pipe(ner, last=True)
  ner.add_label('TOXIC')

  pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
  unaffected_pipes = [
      pipe for pipe in toxic_tagging.pipe_names
      if pipe not in pipe_exceptions]

  print('training')
  with toxic_tagging.disable_pipes(*unaffected_pipes):
    toxic_tagging.begin_training()
    for iteration in range(30):
      random.shuffle(training_data)
      losses = {}
      batches = spacy.util.minibatch(
          training_data, size=spacy.util.compounding(
              4.0, 32.0, 1.001))
      for batch in batches:
        texts, annotations = zip(*batch)
        toxic_tagging.update(texts, annotations, drop=0.5, losses=losses)
      print("Losses", losses)

  # Score on trial data.
  print('evaluation')
  scores = []
  for spans, text in test:
    pred_spans = []
    doc = toxic_tagging(text)
    for ent in doc.ents:
      pred_spans.extend(range(ent.start_char, ent.start_char + len(ent.text)))
    score = semeval2021.f1(pred_spans, spans)
    scores.append(score)
  print('avg F1 %g' % statistics.mean(scores))

#%%

def read_datafile(df):
    df['fixed'] = df.apply(lambda row : fix_spans.fix_spans(row['spans'], row['text']), axis = 1)
    return df[['fixed', 'text']].apply(tuple, axis = 1).to_list()

train = read_datafile(pd.read_pickle('data/train.bin')[['text', 'spans']])
test = read_datafile(pd.read_pickle('data/test.bin')[['text', 'spans']])

main(train, test)

# %%
