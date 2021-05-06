#%%
import os
import re
import json
import string
import numpy as np
import pandas as pd
from ast import literal_eval
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
os.chdir('/home/burtenshaw/now/spans_toxic')
from utils import *
# from toxic_spans.evaluation import fix_spans

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default= '/home/burtenshaw/now/spans_toxic/data/')
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--folds', action='store_true', default=False)
parser.add_argument('--json', action='store_true', default=False)
parser.add_argument('--bin', action='store_true', default=False)
parser.add_argument('--csv', action='store_true', default=False)
args = parser.parse_args()

data_dir = '/home/burtenshaw/now/spans_toxic/data/'
#%%
if args.train:

    df = pd.read_csv(data_dir + "tsd_test.csv")
    df.spans = df.spans.apply(literal_eval)
    data = pd.DataFrame()

    data['text'] = df.text
    data['spans'] = df.spans
    data['ent_check'] = df.spans.apply(lambda e : len(e)-(e[-1]-e[0]) if len(e) > 2 else np.nan)
    data['cont_ranges'] = data.spans.apply(contiguous_ranges)
    data['flat_cont_ranges'] = data.spans.apply(contiguous_ranges, flatten=True)
    #%%
    # spacy 
    print('making spacy doc ... ')
    data['doc'] = data.text.apply(nlp)
    data['entities'] = data.apply( lambda row : spans_to_ents(row.doc, set(row.spans), 'TOXIC'), axis = 1 )
    data['doc_ents'] = data.apply(lambda row : [row.doc.char_span(start, end, label = label) for start, end, label in row.entities ], axis =1)
    data['doc'] = data.apply(add_ents, axis = 1)
    data['word_mask'] = data.doc.apply(spacy_ents_to_word_mask)
    data['tokens'] = data.doc.apply(lambda doc : [token.text for token in doc])
    data.drop(columns = ['doc_ents', 'doc'], inplace = True)
    #%%
    # add baseline values
    print('re - making spacy doc ... ')
    data['doc'] = data.text.apply(nlp)
    print('adding baseline predictions ...')
    data['baseline'] = pd.read_pickle('/home/burtenshaw/now/spans_toxic/data/all/spacy.bin')
    data['baseline_entities'] = data.apply( lambda row : spans_to_ents(row.doc, set(row.baseline), 'TOXIC'), axis = 1 )
    data['doc_ents'] = data.apply(lambda row : [row.doc.char_span(start, end, label = label) for start, end, label in row.baseline_entities ], axis =1)
    data['doc'] = data.apply(add_ents, axis = 1)
    data['baseline_word_mask'] = data.doc.apply(spacy_ents_to_word_mask)

    data.drop(columns = ['doc_ents', 'doc'], inplace = True)
    #%%
    # data.drop(columns = ['baseline', 'baseline_entities'], inplace = True)

    # categorical targets
    print('adding categorical info ...')
    data['n_spans'] = data.cont_ranges.apply(len)
    data['start'] = data.word_mask.apply(lambda x : np.argmax(x) if 1 in x else -1)
    data['end'] = data.word_mask.apply(lambda x : len(x) - np.argmax(x[::-1]) if 1 in x else -1)
    data['len'] = data.apply(lambda x : x.end - x.start if x.start > 0 else -1, axis=1)
    data['tuples'] = data[['word_mask', 'tokens']].apply(lambda row : list(enumerate(zip(row.tokens, row.word_mask))), axis=1)

    #%%
    # examples with toxic spans after max len
    print('examples with toxic spans after max len: ', data.loc[data.word_mask.map(lambda x : 1 in x[128:])].shape[0])

    # check word mask and character entity alignment 
    data['parsing_predictions'] = data.apply(spacy_word_mask_to_spans, axis = 1)
    # data['parsing_predictions'] = data.apply(word_mask_to_character_entity, pad = False, axis = 1)
    print(' parsing accuracy %s' % data.apply(lambda row : \
                                f1(row.parsing_predictions, row.spans), axis = 1).mean())
    # check word mask and token alignment
    print((data.tokens.apply(len) - data.word_mask.apply(len)).describe())

    # %%

    print('saving main split')

    all_path = '%s%s/' % (data_dir, 'all')

    train_index, test_index = train_test_split(data.index, test_size=0.2, random_state=2018)
    train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=2018)

    train = data.loc[train_index]
    val = data.loc[val_index]
    test = data.loc[test_index]

    if args.json:
        data.to_json(data_dir + 'data.json')
        train.to_json(all_path + 'train.json')
        val.to_json(all_path + 'val.json')
        test.to_json(all_path + 'test.json')

    if args.csv:
        data.to_csv(data_dir + 'data.csv')
        train.to_csv(all_path + 'train.csv')
        val.to_csv(all_path + 'val.csv')
        test.to_csv(all_path + 'test.csv')

    if args.bin:
        data.to_pickle(data_dir + 'data.bin')
        train.to_pickle(all_path + 'train.bin')
        val.to_pickle(all_path + 'val.bin')
        test.to_pickle(all_path + 'test.bin')
        
    # %%
    print('folding ...')
    if args.folds:
        kf = KFold(n_splits=5)
        kf.get_n_splits(data.index)

        for n, (train_index, test_index) in enumerate(kf.split(data.index)):
            fold_path = '%s%s/' % (data_dir, n)
            try:
                os.mkdir(fold_path)
            except FileExistsError:
                pass


            train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=2018)
            train, val, test = data.loc[train_index],data.loc[val_index], data.loc[test_index]
            print('%s : ' % n)
            print("TRAIN:", len(train_index), 
                    "VAL:", len(val_index), 
                    "TEST:", len(test_index))

            if args.json:
                train.to_json(fold_path + 'train.json')
                val.to_json(fold_path + 'val.json')
                test.to_json(fold_path + 'test.json')
            
            if args.csv:
                train.to_csv(fold_path + 'train.csv')
                val.to_csv(fold_path + 'val.csv')
                test.to_csv(fold_path + 'test.csv')
            
            if args.bin:
                train.to_pickle(fold_path + 'train.bin')
                val.to_pickle(fold_path + 'val.bin')
                test.to_pickle(fold_path + 'test.bin')

# %%
if args.eval:
    eval_ = pd.read_csv(data_dir + "tsd_test.csv")
    data = pd.DataFrame()
    data['text'] = eval_.text
    data['doc'] = data.text.apply(nlp)
    data['tokens'] = data.doc.apply(lambda doc : [token.text for token in doc])
    data.drop(columns = ['doc'], inplace = True)
    data.to_csv(data_dir + 'eval.csv')
    data.to_pickle(data_dir + 'eval.bin')




