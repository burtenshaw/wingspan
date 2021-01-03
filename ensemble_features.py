#%%
import os
import torch
import logging
import torch
import time
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, roc_curve
from collections import OrderedDict
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from maxfw.model import MAXModelWrapper
os.chdir('/home/burtenshaw/code/dialogue_dash/toxic')
from core.model import ModelWrapper
from core.bert_pytorch import BertForMultiLabelSequenceClassification, InputExample, convert_examples_to_features
from config import DEFAULT_MODEL_PATH, LABEL_LIST, MODEL_META_DATA as model_meta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import spacy

logger = logging.getLogger()

sentence_bert_model = SentenceTransformer('msmarco-distilroberta-base-v2')
analyser = SentimentIntensityAnalyzer()
nlp = spacy.load('en_vectors_web_lg')

#%%
fold = '0'
#%%


def word_level_sentiment(tokens, maxlen = 128):

    ''' Vader Sentiment '''

    wls = np.vstack([np.array(list(analyser.polarity_scores(w).values())) for w in tokens])
    padded_sentiment = np.pad(wls, [(0, maxlen-wls.shape[0]), (0,0)])
    
    return padded_sentiment


def word_vectors(text, maxlen = 128):
    
    ''' spacy vectors [300 dim glove]'''
    
    vectors = np.vstack([w.vector for w in nlp(text[:maxlen])])
    padded_vectors = np.pad(vectors, [(0, maxlen-vectors.shape[0]), (0,0)])
    return padded_vectors


def word_level_features(fold_dir):

    pred_dir = fold_dir
    df = pd.read_pickle(os.path.join(pred_dir, 'test.bin'))

    # spacy glove vectors
    word_level_vectors = df.text.apply(word_vectors).to_list()

    # vader sentiment
    word_level_sentiment = df.tokens.apply(word_level_sentiment).to_list()

    # start
    start_preds = pd.read_pickle(os.path.join(pred_dir, 'start.bin')).values[:,:128]

    # end
    end_preds = pd.read_pickle(os.path.join(pred_dir, 'end.bin')).values[:,:128]

    # len
    len_preds = pd.read_pickle(os.path.join(pred_dir, 'len.bin')).values

    # n_spans
    n_span_preds = pd.read_pickle(os.path.join(pred_dir, 'n_spans.bin')).values

    # bert_span
    span_bert = pd.read_pickle(os.path.join(pred_dir, 'span_bert.bin'))

    # bert_ngram
    ngram_bert = pd.read_pickle(os.path.join(pred_dir, 'ngram_bert.bin'))
    ngram_bert = ngram_bert.groupby(level=0).pred.apply(np.array).apply(pad_mask, max_len =MAX_LEN)

    # lstm_ngram
    ngram_glove_lstm = pd.read_pickle(os.path.join(pred_dir, 'ngram_lstm.bin'))
    ngram_glove_lstm = ngram_glove_lstm.groupby(level=0).pred.apply(np.array).apply(pad_mask, max_len =MAX_LEN)

    _word_level_features = [word_level_vectors,
                            word_level_sentiment,
                            start_preds,
                            end_preds,
                            len_preds,
                            n_span_preds,
                            span_bert,
                            ngram_bert,
                            ngram_glove_lstm]

    word_level_features = np.vstack([np.hstack(features) for features in _word_level_features])

    return word_level_features


class IBMToxicity(ModelWrapper):

    MODEL_META_DATA = model_meta

    def __init__(self, path=DEFAULT_MODEL_PATH):
        """Instantiate the BERT model."""
        logger.info('Loading model from: {}...'.format(path))

        # Load the model
        # 1. set the appropriate parameters
        self.eval_batch_size = 64
        self.max_seq_length = 256
        self.do_lower_case = True

        # 2. Initialize the PyTorch model
        model_state_dict = torch.load(DEFAULT_MODEL_PATH+'pytorch_model.bin', map_location='cpu')
        self.tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_PATH, do_lower_case=self.do_lower_case)
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(DEFAULT_MODEL_PATH,
                                                                                num_labels=len(LABEL_LIST),
                                                                                state_dict=model_state_dict)
        self.device = torch.device('cuda')
        self.model.to(self.device)

        # 3. Set the layers to evaluation mode
        self.model.eval()

        # logger.info('Loaded model')

def sentence_level_sentiment(text):
    return np.array(list(analyser.polarity_scores(text).values()))


def sentence_level_toxic(text_column):
    ibm_model_wrapper = IBMToxicity()
    tox = ibm_model_wrapper.predict(text_column.to_list())
    return [np.array(list(v.values())) for v in tox]


def sentence_level_features(fold_dir):

    pred_dir = fold_dir
    df = pd.read_pickle(os.path.join(pred_dir, 'test.bin'))

    # ibm multilabel toxicity
    ibm_tox = sentence_level_toxic(df.text)

    # vader sentence_level_sentiment
    sentiment = df.text.apply(sentence_level_sentiment).to_list()

    sentence_bert = df.text.apply(sentence_bert_model.encode).to_list()

    features = np.vstack([np.hstack(features) \
                            for features in zip([ibm_tox, sentiment, sentence_bert])])
        
    return features


def all_ensemble_features(folds_dir_list):

    X_word = []
    X_sentence =[]
    y = []

    for fold_dir in folds_dir_list:
        df = pd.read_pickle(os.path.join(fold_dir, 'test.bin'))
        X_word.extend(word_level_features(fold_dir))
        X_sentence.extend(sentence_level_features(fold_dir))
        y.extend(df.word_mask.to_list())

    return [X_word, X_sentence] , y

