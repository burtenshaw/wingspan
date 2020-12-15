import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow_hub as hub
from tensorflow.keras.layers import Layer

import pandas as pd
import numpy as np

from tqdm import *
import os
import io

tf.config.list_physical_devices(device_type='GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def prep_data(data, field):

    max_tags = data[field].apply(len).max()

    X = data.text.values
    y = pad_sequences(data[field], padding='post', maxlen=max_tags, value=0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2018)

    X_mask,X_ids = bert_prep(X_train, max_len=200)

    return X_train, X_test, y_train, y_test, X_mask, X_ids, max_tags


def bert_prep(text, max_len=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    padded_ids = []
    mask_ids = []
    attn_ids = []

    for i in tqdm(range(len(text))):
        encoding = tokenizer.encode_plus(text[i], 
                                         max_length = max_len, 
                                         truncation=True, 
                                         padding='max_length')
        
        padded_ids.append(encoding["input_ids"])
        mask_ids.append(encoding['token_type_ids'])
        attn_ids.append(encoding["attention_mask"])

    x_id = np.array(padded_ids)
    x_mask = np.array(mask_ids)
    x_attn = np.array(attn_ids)
        
    return x_id, x_mask, x_attn

def build_bert(input_dim = 200, output_dim=6, dropout=0.2):
    
    input_1 = tf.keras.Input(shape = (input_dim) , dtype=np.int32)
    input_2 = tf.keras.Input(shape = (input_dim) , dtype=np.int32)
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    output  = model([input_1 , input_2] , training = True )
    output = tf.keras.layers.Dropout(dropout)(output[0])
    answer = tf.keras.layers.Dense(output_dim, activation = tf.nn.sigmoid )(output)

    logits = tf.keras.layers.Flatten()(answer)
    probs = tf.keras.layers.Activation(tf.keras.activations.softmax)(logits)
    
    answer = tf.keras.layers.Dense(output_dim, activation = tf.nn.sigmoid )(probs)

    model = tf.keras.Model(inputs = [input_1, input_2 ] , outputs = [answer])

    model.summary()

    auc_score = AUC(multi_label=True)
    
    model.compile(optimizer = Adam(lr = 3e-5),
                  loss = tf.keras.losses.binary_crossentropy,
                  metrics = [ auc_score])
    return model


def load_vec(emb_path_list, nmax=50000):
    words = []
    embeddings_index = {}
    for p_n, emb_path in enumerate(emb_path_list):
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                if word in words:
                    break
                else:
                    vect = np.fromstring(vect, sep=' ')
                    embeddings_index[word] = vect
                if len(words) == nmax*p_n:
                    break
    return embeddings_index, words

def muse_embedding(X_train):
    src_path = '/home/burtenshaw/code/toxic_direction/toxicity/multilabel_classification/vectors/wiki.multi.en.vec'
    tgt_path = '/home/burtenshaw/code/toxic_direction/toxicity/multilabel_classification/vectors/wiki.multi.nl.vec'
    nmax = 50000  # maximum number of word embeddings to load

    embeddings_index, words = load_vec([src_path,tgt_path], nmax)

    vectorizer = TextVectorization(max_tokens=80000, output_sequence_length=200)
    text_ds = tf.data.Dataset.from_tensor_slices(list(X_train) + words).batch(128)
    vectorizer.adapt(text_ds)

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    num_tokens = len(voc) + 2
    embedding_dim = 300
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word.decode())
    #     print(embedding_vector)
    #     break
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix, num_tokens, vectorizer

def muse_prep(X_train, X_test, y_train, y_test, vectorizer):
    x_train = vectorizer(np.array([[s] for s in X_train])).numpy()
    x_val = vectorizer(np.array([[s] for s in X_test])).numpy()
    y_train = np.array(y_train)
    y_val = np.array(y_test)
    return x_train, x_val, y_train, y_val

def build_muse_lstm(embedding_matrix, num_tokens, output_dim=6):
    deep_inputs = tf.keras.Input(shape=(200,))
    embedding_layer = layers.Embedding(num_tokens, 300, 
                                weights=[embedding_matrix], 
                                trainable=False)(deep_inputs)
    # LSTM_Layer_1 = layers.LSTM(128)(embedding_layer)
    bi_lstm_1 = layers.Bidirectional(layers.LSTM(150, return_sequences=True))(embedding_layer)
    bi_lstm_2 = layers.Bidirectional(layers.LSTM(150))(bi_lstm_1)
    dense_layer_1 = layers.Dense(output_dim, activation='sigmoid')(bi_lstm_2)
    model = tf.keras.Model(inputs=deep_inputs, outputs=dense_layer_1)
    auc_score = AUC(multi_label=True, curve='PR')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_score])
    return model


def dual_bert():

    '''
    https://github.com/cerlymarco/MEDIUM_NoteBook/blob/master/Siamese_Dual_BERT/Siamese_Dual_BERT.ipynb
    '''
    
    opt = Adam(learning_rate=2e-5)
    
    id1 = tf.keras.layers.Input((128,), dtype=tf.int32)
    mask1 = tf.keras.layers.Input((128,), dtype=tf.int32)
    atn1 = tf.keras.layers.Input((128,), dtype=tf.int32)
    
    id2 = tf.keras.layers.Input((1,), dtype=tf.int32)
    mask2 = tf.keras.layers.Input((1,), dtype=tf.int32)
    atn2 = tf.keras.layers.Input((1,), dtype=tf.int32)
    
    
    config = BertConfig() 
    config.output_hidden_states = False # Set to True to obtain hidden states
    bert_model1 = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    bert_model2 = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    embedding1 = bert_model1(id1, attention_mask=mask1, token_type_ids=atn1)[0]
    embedding2 = bert_model2(id2, attention_mask=mask2, token_type_ids=atn2)[0]
    
    x1 = tf.keras.layers.GlobalAveragePooling1D()(embedding1)
    x2 = tf.keras.layers.GlobalAveragePooling1D()(embedding2)
    
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[id1, mask1, atn1, id2, mask2, atn2], outputs=out)
    return model