#%%
import pandas as pd
import spacy
from spacy.tokens import Span
import os
os.chdir('/home/burtenshaw/now/spans_toxic')

nlp = spacy.load("en_core_web_sm")

data = pd.read_pickle('data/train.bin')
play = data[:20]
# %%
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

  # ents = [Span(doc,start,end, label=label) for start, end, label in ents]
  # doc.ents = ents
  return ents

#%%
play.apply( lambda row : spans_to_ents(nlp(row.text), \
                         set(row.entities), 'TOXIC'), axis = 1 )
# %%
