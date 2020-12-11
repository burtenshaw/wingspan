import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from string import punctuation
from models import bert_prep
from sklearn.model_selection import train_test_split


# PREPROCESSING / MANIPULATION

def word_mask_to_character_entity(row):
    """
    Get the offsets of the toxic spans
    WARNING: Valid only for empty space tokenisation.
    :return: a list with offsets found toxic
    """
    current_offset = 0
    toxic_offsets = []
    for n, word in enumerate(row.text.split(' ')):
        if row.word_mask[n] == 1:
            toxic_offsets.extend(list(range(current_offset, current_offset+len(word))))
        current_offset += len(word) + 1
    return toxic_offsets


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

def mask_to_entities(target_mask):
    return np.where(np.array(target_mask) == 1)[0]


# training  

def prep_data(data, field):

    max_tags = data[field].apply(len).max()

    X = data.text.values
    y = pad_sequences(data[field], padding='post', maxlen=max_tags, value=0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2018)

    X_mask,X_ids = bert_prep(X_train, max_len=200)

    return X_train, X_test, y_train, y_test, X_mask, X_ids, max_tags

# eval

def mask_to_entities(mask):
    mask = np.where(mask.values == 1)[0][:200]
    return np.pad(mask, (0,200-mask.shape[0]))

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

