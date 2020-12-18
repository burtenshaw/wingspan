import numpy as np
from string import punctuation
from models import bert_prep
from sklearn.model_selection import train_test_split
import itertools

from tensorflow.keras.utils import to_categorical

# PREPROCESSING / MANIPULATION

def word_mask_to_character_entity(row, field = 'word_mask', result_shape=200, pad = True):
    """
    Get the offsets of the toxic spans
    WARNING: Valid only for empty space tokenisation.
    :return: a list with offsets found toxic
    """
    current_offset = 0
    toxic_offsets = []
    if type(row.text) != str:
        row.text = row.text.apply(lambda x : x[0])
    for n, word in enumerate(row.text.split(' ')):
        if row[field][n] == 1:
            toxic_offsets.extend(list(range(current_offset, current_offset+len(word))))
        current_offset += len(word) + 1
    if pad:
        mask = np.array(toxic_offsets)
        toxic_offsets = np.pad(mask, (0, result_shape-mask.shape[0]))
    return toxic_offsets

def word_mask_to_spans(row):
    """
    Replication of the above. could be used to improve parsin accuracy
    """

    cont_ranges = []
    prev_char_pos = 0
    prev_binary = mask[0]
    for n, span in itertools.groupby(zip(row.mask, row.tokens), lambda p: len(p[1]) - p[0]):
        now_binary, word = list(span)[0]
        now_char_pos = len(word) + 1 + prev_char_pos

        if prev_binary == 0 and now_binary == 1:
            cont_ranges.extend(range(prev_char_pos, now_char_pos))
            on_span = n
        
        elif prev_binary == 1 and now_binary == 1:
            cont_ranges.extend(range(prev_char_pos - (n-on_span), now_char_pos - 1))

        prev_binary = now_binary
        prev_char_pos = now_char_pos

    return cont_ranges

def contiguous_ranges(span_list, flatten=False):
    """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)].
    From: https://github.com/ipavlopoulos/toxic_spans/blob/master/evaluation/fix_spans.py"""
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    
    if flatten:
        flatten = lambda t: [item for sublist in t for item in sublist]
        return flatten(output)
    return output


def make_character_mask(entities, max_len=1000):
    mask = np.zeros(max_len, int)
    for char_id in entities:
        mask[char_id] = 1 
    return mask


def make_word_mask(row):
    tags = []
    char_pos = 0
    for n, word in enumerate(row.text.split(' ')):
        if row.character_mask[char_pos] == 1:
            tags.append(1)
        elif row.character_mask[char_pos] == 0:
            tags.append(0)

        char_pos += len(word) + 1

    return tags 

def character_mask_to_entities(target_mask):
    return np.where(np.array(target_mask) == 1)[0]

def pad_mask(mask, max_len = 200):
    mask = mask[:max_len]
    return np.pad(mask, (0,max_len - mask.shape[0]))

def word_id_to_mask(word_id, max_len = 200):
    return pad_mask(to_categorical(word_id), max_len = max_len)

def mask_to_entities(mask):
    mask = np.where(mask.values == 1)[0][:200]
    return np.pad(mask, (0,200-mask.shape[0]))

def make_word_level_df(train, max_len = 200):
    return train[['text', 'tuples']].explode('tuples').apply(\
        lambda x : {'text' : x.text, 
                    'x_word_mask': word_id_to_mask(x.tuples[0], max_len = max_len), 
                    'word' : x.tuples[1][0], 
                    'label' : x.tuples[1][1]},\
        result_type = 'expand', axis = 1)

def make_context_labelling(row, pre = 2, post = 2, word = 0):
    context_label = []
    for n, _label in enumerate(row.word_mask):
        start = n-pre
        if start < 0:
            start = 0
        _pre = ' '.join(row.tokens[start:n])
        _word = row.tokens[n]
        _post = ' '.join(row.tokens[n+1:n+post+1])
        
        if word:
            context_label.append((_pre, _word, _post, _label))
        else:
            context_label.append((_pre + ' ' + _word, ' ', _post, _label))

    return context_label




# EVALUATION SPECIFIC TO SEMEVAL 2020

# f1 = 2*(Recall * Precision) / (Recall + Precision)
# def f1(predictions, gold):
#    rec = recall(predictions, gold)
#    prec = precision(predictions,gold)
#    return 0 if (rec + prec == 0) else (2*(rec * prec) / (rec + prec))

def precision(predictions, gold): # TP/TP+FP
    TP = len(set(predictions).intersection(set(gold)))
    FP = len(set(predictions) - set(gold))
    return 0 if (TP+FP==0) else TP / (TP+FP)

def recall(predictions, gold): # TP/TP+FN
    TP = len(set(predictions).intersection(set(gold)))
    FN = len(set(gold) - set(predictions))
    return 0 if (TP+FN==0) else TP / (TP+FN)

def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions)==0 else 0
    nom = 2*len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions))+len(set(gold))
    return nom/denom

def pairwise_operator(codes, method):
    """
    Pairwsise operator based on a method and a list of predictions (e.g., lists of offsets)
    >>> assert pairwise_operator([[],[],[]], f1) == 1
    :param codes: a list of lists of predicted offsets
    :param method: a method to use to compare all pairs
    :return: the mean score between all possible pairs (excl. duplicates)
    """
    pairs = []
    for i,coderi in enumerate(codes):
        for j,coderj in enumerate(codes):
            if j>i:
                pairs.append(method(coderi, coderj))
    return np.mean(pairs)