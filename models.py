import pandas as pd
import numpy as np
import os
import io
from tqdm import *
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras as K
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split


def make_context_labelling(row, pre = 2, post = 2, word = 0):
    
    context_label = []
    for n, _label in enumerate(row.word_mask):
        start = n-pre
        if start < 0:
            start = 0
        _pre = row.sequences[start:n+1]
        _word = [row.sequences[n]]
        _post = row.sequences[n+1:n+post+1]
        
        if word:
            context_label.append({'pre' : _pre, 
                                  'word' : _word, 
                                  'post' : _post, 
                                  'label' : _label})
        else:
            _pre.extend(_word)
            context_label.append({'pre' : _pre, 
                                  'word' : [], 
                                  'post' : _post, 
                                  'label' : _label})

    return context_label

def make_context_data(data, pre = 2, post = 2, word = 1):
    pad = lambda sequences, maxlen: pad_sequences(sequences, maxlen=maxlen)
    
    X_y = data.apply(make_context_labelling, axis = 1, pre = pre, post = post, word = word)\
        .explode().dropna().apply(pd.Series)

    X = [pad(X_y.pre.values, pre),
         pad(X_y.word.values, word),
         pad(X_y.post.values, post)]
    
    y = X_y.label.values

    return X, y, X_y.index

def make_BERT_context_labelling(row, pre = 2, post = 2, word = 0):
    context_label = []
    for n, _label in enumerate(row.word_mask):
        start = n-pre
        if start < 0:
            start = 0

        pre_input_ids = [row.input_ids[0]] + row.input_ids[start:n+1]
        pre_token_type_ids = [row.token_type_ids[0]] + row.token_type_ids[start:n+1]
        pre_attn_mask = [row.attn_mask[0]] + row.attn_mask[start:n+1]

        post_input_ids = [row.input_ids[0]] + row.input_ids[n:n+post+1]
        post_token_type_ids = [row.token_type_ids[0]] + row.token_type_ids[n:n+post+1]
        post_attn_mask = [row.attn_mask[0]] + row.attn_mask[n:n+post+1]

        context_label.append({'pre_input_ids' : pre_input_ids,
                              'pre_token_type_ids' : pre_token_type_ids,
                              'pre_attn_mask' : pre_attn_mask,
                              'post_input_ids' : post_input_ids,
                              'post_token_type_ids' : post_token_type_ids,
                              'post_attn_mask' : post_attn_mask, 
                              'label' : _label})

    return context_label

def make_BERT_context_data(data, pre = 2, word = 0, post = 2):

    X_y = data.apply(make_BERT_context_labelling, axis = 1, pre = pre, post = post, word = word)\
        .explode().dropna().apply(pd.Series)

    X = [pad_sequences(X_y.pre_input_ids.values, maxlen = pre+1 ),
         pad_sequences(X_y.pre_token_type_ids.values, maxlen = pre+1 ),
         pad_sequences(X_y.pre_attn_mask.values, maxlen = pre+1 ),
         pad_sequences(X_y.post_input_ids.values, maxlen = post+1 ),
         pad_sequences(X_y.post_token_type_ids.values, maxlen = post+1 ),
         pad_sequences(X_y.post_attn_mask.values, maxlen = post+1 )]

    y = X_y.label.astype(np.int).values

    return X, y, X_y.index

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

    input_ids = np.array(padded_ids)
    token_type_ids = np.array(mask_ids)
    attn_mask = np.array(attn_ids)
        
    return input_ids, token_type_ids, attn_mask

# used in siamese lstm

def get_embedding_weights(word_index , src_path = '/home/corpora/word_embeddings', embedding = 'glove.6B.100d.txt'):

    embeddings_index = {}
    f = open(os.path.join(src_path, embedding))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_class_weights(labels):
    neg, pos = np.bincount(labels.astype(np.int64))
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return class_weight


def ngram_dual_bert(data, pre_length, post_length, hparams, callbacks, metrics, return_model = False): 

    pre_ids = tf.keras.layers.Input((pre_length,), dtype=tf.int32)
    pre_tok_types = tf.keras.layers.Input((pre_length,), dtype=tf.int32)
    pre_attn_mask = tf.keras.layers.Input((pre_length,), dtype=tf.int32)
    
    post_ids = tf.keras.layers.Input((post_length,), dtype=tf.int32)
    post_tok_types = tf.keras.layers.Input((post_length,), dtype=tf.int32)
    post_attn_mask = tf.keras.layers.Input((post_length,), dtype=tf.int32)
    
    config = BertConfig() 
    config.output_hidden_states = False # Set to True to obtain hidden states
    
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    pre_embedded = bert_model(pre_ids, attention_mask=pre_attn_mask, token_type_ids=pre_tok_types)[0]
    post_embedded = bert_model(post_ids, attention_mask=post_attn_mask, token_type_ids=post_tok_types)[0]
    
    input_length = pre_length + post_length
    model_scale = hparams['model_scale']

    if hparams['lstm']:
        merged = tf.keras.layers.concatenate([pre_embedded, post_embedded], axis=1)
        layer =  layers.Bidirectional(layers.LSTM(input_length*model_scale))(merged)
    else:
        pre_embedded = tf.keras.layers.GlobalAveragePooling1D()(pre_embedded)
        post_embedded = tf.keras.layers.GlobalAveragePooling1D()(post_embedded)
        merged = tf.keras.layers.concatenate([pre_embedded, post_embedded], axis=1)
        layer = layers.Dense(input_length*hparams['model_scale'], activation=hparams['activation'])(merged)
        
    for _ in range(hparams['n_layers']):
        layer = layers.Dense(input_length*hparams['model_scale'], activation=hparams['activation'])(layer)
        layer = tf.keras.layers.Dropout(hparams['dropout'])(layer)
        model_scale = model_scale / 2

    out = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
    
    model = tf.keras.Model(inputs=[pre_ids, 
                                   pre_tok_types, 
                                   pre_attn_mask, 
                                   post_ids, 
                                   post_tok_types, 
                                   post_attn_mask], 

                           outputs=out)

    opt = Adam(lr = hparams['lr'])

    model.compile(optimizer = opt, 
                  loss = 'binary_crossentropy', 
                  metrics = metrics)

    class_weight = get_class_weights(data['y_train'])

    model.fit(  data['X_train'] , 
                data['y_train'],
                batch_size=hparams['batch_size'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=hparams['epochs'],
                verbose = 1,
                callbacks= callbacks,
                class_weight = class_weight)

    scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)

    if return_model:
        return model
    else:
        scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
        return scores

def ngram_glove_lstm(data, pre_length, word_length, post_length, hparams, callbacks, metrics, embedding_matrix = 0, return_model = False):  

    pre = tf.keras.Input(shape=(pre_length,), dtype="int64")
    word = tf.keras.Input(shape=(word_length,), dtype="int64")
    post = tf.keras.Input(shape=(post_length,), dtype="int64")
    
    pre_embedding = layers.Embedding(embedding_matrix.shape[0], 
                            100, 
                            weights=[embedding_matrix],
                            input_length = pre_length, trainable=True)

    word_embedding = layers.Embedding(embedding_matrix.shape[0], 
                        100, 
                        weights=[embedding_matrix],
                        input_length = word_length, trainable=True)

    post_embedding = layers.Embedding(embedding_matrix.shape[0], 
                        100, 
                        weights=[embedding_matrix],
                        input_length = post_length, trainable=True)
    
    pre_embedded = pre_embedding(pre)
    word_embedded = word_embedding(word)
    post_embedded = post_embedding(post)

    merged = tf.keras.layers.concatenate([pre_embedded, word_embedded, post_embedded], axis=1)
    input_length = pre_length + word_length + post_length

    layer =  layers.Bidirectional(layers.LSTM(input_length*hparams['model_scale']))(merged)

    model_scale = hparams['model_scale']

    for _ in range(hparams['n_layers']):
        layer = layers.Dense(input_length*hparams['model_scale'], activation=hparams['activation'])(layer)
        layer = tf.keras.layers.Dropout(hparams['dropout'])(layer)
        model_scale = model_scale / 2

    output = layers.Dense(1, activation='sigmoid')(layer)

    model = tf.keras.Model(
        inputs=[pre, word, post],
        outputs=[output],
    )

    opt = Adam(lr = hparams['lr'])

    model.compile(optimizer = opt, 
                  loss = 'binary_crossentropy', 
                  metrics = metrics)

    class_weight = get_class_weights(data['y_train'])

    model.fit(  data['X_train'] , 
                data['y_train'],
                batch_size=hparams['batch_size'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=hparams['epochs'],
                verbose = 1,
                callbacks= callbacks,
                class_weight = class_weight)

    if return_model:
        return model
    else:
        scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
        return scores

# do siamese lstm as class 


def ngram_single_bert(data, input_length, hparams, callbacks, metrics, embedding_matrix = 0): 

    ids = tf.keras.layers.Input((input_length,), dtype=tf.int32)
    tok_types = tf.keras.layers.Input((input_length,), dtype=tf.int32)
    attn_mask = tf.keras.layers.Input((input_length,), dtype=tf.int32)
    
    config = BertConfig() 
    config.output_hidden_states = False # Set to True to obtain hidden states
    
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    embedded = bert_model(ids, attention_mask=attn_mask, token_type_ids=tok_types)[0]
    
    model_scale = hparams['model_scale']

    layer =  layers.LSTM(input_length*model_scale)(embedded)

    for _ in range(hparams['n_layers']):
        layer = layers.Dense(input_length*hparams['model_scale'], activation=hparams['activation'])(layer)
        layer = tf.keras.layers.Dropout(hparams['dropout'])(layer)
        model_scale = model_scale / 2

    out = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
    
    model = tf.keras.Model(inputs=[ids, 
                                   tok_types, 
                                   attn_mask], 

                           outputs=out)

    model.compile(optimizer = Adam(lr = hparams['lr']), 
                  loss = 'binary_crossentropy', 
                  metrics = metrics)

    class_weight = get_class_weights(data['y_train'])

    model.fit(data['X_train'] , 
              data['y_train'],
              batch_size=hparams['batch_size'],
              validation_data=(data['X_val'], data['y_val']),
              epochs=hparams['epochs'],
              verbose = 1,
              callbacks= callbacks,
              class_weight = class_weight)

    scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)

    return scores

def bert_to_mask(data, input_length, output_length, hparams, callbacks, metrics, loss = 'categorical_crossentropy', embedding_matrix = 0, return_model = False): 

    ids = tf.keras.layers.Input((input_length,), dtype=tf.int32)
    tok_types = tf.keras.layers.Input((input_length,), dtype=tf.int32)
    attn_mask = tf.keras.layers.Input((input_length,), dtype=tf.int32)
    
    config = BertConfig() 
    config.output_hidden_states = False # Set to True to obtain hidden states
    
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    embedded = bert_model(ids, attention_mask=attn_mask, token_type_ids=tok_types)[0]
    
    model_scale = hparams['model_scale']

    layer =  layers.Bidirectional(layers.LSTM(input_length*model_scale))(embedded)

    for _ in range(hparams['n_layers']):
        layer = layers.Dense(input_length*hparams['model_scale'], activation=hparams['activation'])(layer)
        layer = tf.keras.layers.Dropout(hparams['dropout'])(layer)
        model_scale = model_scale / 2

    out = tf.keras.layers.Dense(output_length, activation='softmax')(layer)
    
    model = tf.keras.Model(inputs=[ids, 
                                   tok_types, 
                                   attn_mask], 

                           outputs=out)

    model.compile(optimizer = Adam(lr = hparams['lr']), 
                  loss = loss, 
                  metrics = metrics)

    model.fit(data['X_train'] , 
              data['y_train'],
              batch_size=hparams['batch_size'],
              validation_data=(data['X_val'], data['y_val']),
              epochs=hparams['epochs'],
              verbose = 1,
              callbacks= callbacks)

    if return_model:
        return model
    else:
        scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
        return scores

def categorical_bert(data, input_length, output_length, hparams, callbacks, metrics, verbose = 1, loss = 'categorical_crossentropy', embedding_matrix = 0, return_model = False): 

    ids_input = layers.Input((input_length,), dtype=tf.int32)
    token_type_input = layers.Input((input_length,), dtype=tf.int32)
    attn_mask_input = layers.Input((input_length,), dtype=tf.int32)
    
    config = BertConfig() 
    config.output_hidden_states = False # Set to True to obtain hidden states
    
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    embedded = bert_model(ids_input, attention_mask=attn_mask_input, token_type_ids=token_type_input)[0]
    
    model_scale = hparams['model_scale']

    layer =  layers.Dense(input_length*model_scale)(embedded)
    layer = layers.Dense(output_length)(layer)
    layer = tf.keras.layers.GlobalAveragePooling1D()(layer)

    out = layers.Activation(K.activations.softmax)(layer)

    model = tf.keras.Model(inputs=[ids_input, 
                                   token_type_input, 
                                   attn_mask_input], 
                           outputs=out)

    # loss = K.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer = Adam(lr = hparams['lr']), 
                  loss = loss, 
                  metrics = metrics)

    model.fit(data['X_train'], 
              data['y_train'],
              batch_size=hparams['batch_size'],
              validation_data=(data['X_val'], data['y_val']),
              epochs=hparams['epochs'],
              verbose = verbose,
              callbacks= callbacks)

    if return_model:
        return model
    else:
        scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
        return scores



''' LEGACY FUNCTIONS '''



def prep_data(data, field):

    max_tags = data[field].apply(len).max()

    X = data.text.values
    y = pad_sequences(data[field], padding='post', maxlen=max_tags, value=0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2018)

    X_mask,X_ids = bert_prep(X_train, max_len=200)

    return X_train, X_test, y_train, y_test, X_mask, X_ids, max_tags

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

def muse_embedding(X_train, output_sequence_length=200):
    src_path = '/home/corpora/word_embeddings/fasttext.wiki.en.txt'
    nmax = 50000  # maximum number of word embeddings to load

    embeddings_index, words = load_vec([src_path], nmax)

    vectorizer = TextVectorization(max_tokens=80000, output_sequence_length=output_sequence_length)
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
        embedding_vector = embeddings_index.get(word)
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