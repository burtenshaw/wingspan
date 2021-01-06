#%%
import os
import torch
import logging
import torch
import time
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, roc_curve
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import spacy

#%%
sentence_bert_model = SentenceTransformer('msmarco-distilroberta-base-v2')
analyser = SentimentIntensityAnalyzer()
nlp = spacy.load('en_vectors_web_lg')


BASE_DIR = '/home/burtenshaw/now/spans_toxic'
os.chdir(BASE_DIR)

from utils import pad_mask
fold = '0'
#%%

def word_sentiment(tokens, maxlen = 128):

    ''' Vader Sentiment '''

    wls = np.vstack([np.array(list(analyser.polarity_scores(w).values())) for w in tokens[:maxlen]])
    wls = np.pad(wls, [(0, maxlen-wls.shape[0]), (0,0)])
    
    return wls


def word_vectors(text, maxlen = 128):
    
    ''' spacy vectors [300 dim glove]'''
    
    vectors = np.vstack([w.vector for w in nlp(text[:maxlen])])
    padded_vectors = np.pad(vectors, [(0, maxlen-vectors.shape[0]), (0,0)])
    return padded_vectors


def word_level_features(fold_n, maxlen = 128):

    pred_dir = os.path.join(BASE_DIR, 'predictions', fold_n)
    data_dir = os.path.join(BASE_DIR, 'data', fold_n)

    df = pd.read_pickle(os.path.join(data_dir, 'test.bin'))

    # spacy glove vectors
    word_level_vectors = df.text.apply(word_vectors).to_list()

    # vader sentiment
    word_level_sentiment = df.tokens.apply(word_sentiment).to_list()

    # start
    start_preds = list(pd.read_pickle(os.path.join(pred_dir, 'start.bin')).values[:,:maxlen])

    # end
    end_preds = list(pd.read_pickle(os.path.join(pred_dir, 'end.bin')).values[:,:maxlen])

    # bert_span
    span_bert = list(pd.read_pickle(os.path.join(pred_dir, 'bert_span.bin')).values[:,:maxlen])

    # bert_ngram
    bert_ngram = pd.read_pickle(os.path.join(pred_dir, 'bert_ngram.bin'))
    bert_ngram = bert_ngram.groupby(level=0)[0].apply(np.array)\
                    .apply(pad_mask, max_len =maxlen).to_list()

    # lstm_ngram
    ngram_lstm = pd.read_pickle(os.path.join(pred_dir, 'lstm_ngram.bin'))
    ngram_lstm = ngram_lstm.groupby(level=0)[0].apply(np.array)\
                    .apply(pad_mask, max_len =maxlen).to_list()
    
    model_predictions = [start_preds, end_preds, span_bert, bert_ngram, ngram_lstm]
    model_predictions = [np.vstack(p).T for p in zip(*model_predictions)]

    _word_level_features = [word_level_vectors,
                            word_level_sentiment,
                            model_predictions]

    word_level_features = [np.hstack(features).T for features in zip(*_word_level_features)]

    return word_level_features

def sentence_level_sentiment(text):
    return np.array(list(analyser.polarity_scores(text).values()))


def sentence_level_toxic(ibm_preds):
    return ibm_preds


def sentence_level_features(fold):

    pred_dir = os.path.join(BASE_DIR, 'predictions', fold)
    data_dir = os.path.join(BASE_DIR, 'data', fold)

    df = pd.read_pickle(os.path.join(data_dir, 'test.bin'))

    scaler = MaxAbsScaler()

    # len
    len_preds = pd.read_pickle(os.path.join(pred_dir, 'len.bin')).values
    len_preds = scaler.fit_transform(np.argmax(len_preds, -1).reshape(-1,1))

    # n_spans
    n_span_preds = pd.read_pickle(os.path.join(pred_dir, 'n_spans.bin')).values
    n_span_preds = scaler.fit_transform(np.argmax(n_span_preds, -1).reshape(-1,1))

    # ibm multilabel toxicity
    ibm_tox = np.vstack(pd.read_pickle(os.path.join(pred_dir, 'ibm_tox.bin')).values)

    # vader sentence_level_sentiment
    sentiment = np.vstack(df.text.apply(sentence_level_sentiment).values)

    sentence_bert = np.vstack(df.text.apply(sentence_bert_model.encode).values)
    
    _features = [len_preds,
                 n_span_preds,
                 ibm_tox,
                 sentiment,
                 sentence_bert]

    features = [x.reshape(1,-1) for x in list(np.hstack(_features))]
     
    return features

#%%

def all_ensemble_features(folds_dir_list):

    X_word = []
    X_sentence =[]
    y = []
    text = []
    spans = []

    for fold_dir in folds_dir_list:
        df = pd.read_pickle(os.path.join(BASE_DIR, 'data', fold_dir, 'test.bin'))
        X_word.extend(word_level_features(fold_dir))
        X_sentence.extend(sentence_level_features(fold_dir))
        y.extend(df.word_mask.apply(pad_mask, max_len = 128).to_list())
        text.extend(df.text.to_list())
        spans.extend(df.spans.to_list())

    return X_word, X_sentence , y, text, spans


# %%

def get_data(folds_dir_list, limit = None, test_text = False):

    X_word, X_sentence , y, text, spans = all_ensemble_features(folds_dir_list)

    if limit:
        X_word, X_sentence , y = X_word[:limit], X_sentence[:limit] , y[:limit]
    
    X_w_train, X_w_test, y_train, y_test = train_test_split(X_word, y, test_size=0.2, shuffle=False)
    X_s_train, X_s_test, _, _ = train_test_split(X_sentence, y, test_size=0.2, shuffle=False)
    


    X_train = [np.dstack(X_w_train).T, np.dstack(X_s_train).T]
    y_train = np.vstack(y_train)
    X_test = [np.dstack(X_w_test).T, np.dstack(X_s_test).T]
    y_test = np.vstack(y_test)

    if test_text:
        _, test_text, _, test_spans = train_test_split(text, spans, test_size=0.2, shuffle=False)
        return X_train, y_train, X_test, y_test, [test_text, test_spans]
    else:
        return X_train, y_train, X_test, y_test

# %%
