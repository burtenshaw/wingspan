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
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split


class SiameseBert:

    def __init__(self, train, val, test, maxlen = 200):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen= maxlen
        self.add_bert_sequences()

        self.metrics = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            AUC(name='auc')
        ]

    def get_data(self):

        self.pre = self.hparams['pre']
        self.post = self.hparams['post']

        X_train, y_train, self.train_index = self.make_BERT_context_data(\
                            self.train, pre = self.pre, post = self.post)
        X_val, y_val, self.val_index = self.make_BERT_context_data(\
                            self.val, pre = self.pre, post = self.post)
        X_test, y_test, self.test_index = self.make_BERT_context_data(\
                            self.test, pre = self.pre, post = self.post)

        train_samples = {'X_train' : X_train, 
                'y_train' : y_train, 
                'X_val' : X_val, 
                'y_val' : y_val, 
                'X_test' : X_test, 
                'y_test' : y_test}


        return train_samples

    def add_bert_sequences(self):
        
        self.val['input_ids'], self.val['token_type_ids'], self.val['attn_mask'] = \
            [x.tolist() for x in bert_prep(self.val.text.to_list(), max_len = self.maxlen)]
        self.train['input_ids'], self.train['token_type_ids'], self.train['attn_mask'] = \
            [x.tolist() for x in bert_prep(self.train.text.to_list(), max_len = self.maxlen)]
        self.test['input_ids'], self.test['token_type_ids'], self.test['attn_mask'] = \
            [x.tolist() for x in bert_prep(self.test.text.to_list(), max_len = self.maxlen)]

    def make_BERT_context_labelling(self, row, pre = 2, post = 2, word = 0):
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

    def make_BERT_context_data(self, data, pre = 2, word = 0, post = 2):

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

    def run(self, data, return_model = False): 
        
        pre_length = self.pre + 1 
        post_length = self.post + 1
        hparams = self.hparams
        
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
                    metrics = self.metrics)

        class_weight = get_class_weights(data['y_train'])

        model.fit(  data['X_train'] , 
                    data['y_train'],
                    batch_size=hparams['batch_size'],
                    validation_data=(data['X_val'], data['y_val']),
                    epochs=hparams['epochs'],
                    verbose = 1,
                    callbacks= self.callbacks,
                    class_weight = class_weight)

        scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)

        if return_model:
            return model
        else:
            scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
            return scores

class CategoricalBert:

    def __init__(self, train, val, test, hparams, method_name, max_len, output_len):

        self.train = train
        self.val = val
        self.test = test
        self.hparams = hparams
        self.method_name = method_name
        self.max_len = max_len
        self.output_len = output_len

    
    def bert_prep(self, text, max_len=128):
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

    def get_data(self):

        X_train = self.bert_prep(train.text.to_list(), max_len = self.max_len)
        y_train = to_categorical(train[self.method_name].values, num_classes = self.output_len)

        X_val = self.bert_prep(val.text.to_list(), max_len = self.max_len)
        y_val = to_categorical(val[self.method_name].values, num_classes = self.output_len)

        X_test = self.bert_prep(test.text.to_list(), max_len = self.max_len)
        y_test = to_categorical(test[self.method_name].values, num_classes = self.output_len)

        train_samples = {'X_train' : X_train, 
                    'y_train' : y_train, 
                    'X_val' : X_val, 
                    'y_val' : y_val, 
                    'X_test' : X_test, 
                    'y_test' : y_test}

        return train_samples

    def run(self, data, input_length, output_length, hparams, callbacks, metrics, verbose = 1, loss = 'categorical_crossentropy', embedding_matrix = 0, return_model = False): 

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


class SiameseLSTM:

    def __init__(self, train, val, test):

        self.train = train
        self.val = val
        self.test = test

        self.make_vocab()

    def get_data(self):

        self.pre = self.hparams['pre']
        self.word = self.hparams['word']
        self.post = self.hparams['post']

        X_train, y_train, self.train_index = make_context_data(train, 
                                                pre = self.pre, 
                                                post = self.post, 
                                                word = self.word)
        
        X_val, y_val, self.val_index = make_context_data(val, 
                                            pre = self.pre, 
                                            post = self.post, 
                                            word = self.word)
        
        X_test, y_test, self.test_index = make_context_data(test, 
                                              pre = self.pre, 
                                              post = self.post, 
                                              word = self.word)

        train_samples = {'X_train' : X_train, 
                        'y_train' : y_train, 
                        'X_val' : X_val, 
                        'y_val' : y_val, 
                        'X_test' : X_test, 
                        'y_test' : y_test}

        return train_samples


    def make_vocab(self):
        index_word = dict(enumerate(set(self.train.tokens.explode().to_list() + \
                                        self.val.tokens.explode().to_list() + \
                                        self.test.tokens.explode().to_list())))

        word_index =  dict(map(reversed, index_word.items()))

        self.train['sequences'] = self.train.tokens.apply(lambda sentence :\
                                 [word_index[w] for w in sentence]).to_list()
        self.test['sequences'] = self.test.tokens.apply(lambda sentence :\
                                 [word_index[w] for w in sentence]).to_list()
        self.val['sequences'] = self.val.tokens.apply(lambda sentence :\
                                 [word_index[w] for w in sentence]).to_list()

        self.embedding_matrix = self.get_embedding_weights(word_index)
        self.word_index = word_index
        self.index_word = index_word


    def make_context_labelling(self, row, pre = 2, post = 2, word = 0):
        
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

    def make_context_data(self, data, pre = 2, post = 2, word = 1):
        pad = lambda sequences, maxlen: pad_sequences(sequences, maxlen=maxlen)
        
        X_y = data.apply(make_context_labelling, axis = 1, pre = pre, post = post, word = word)\
            .explode().dropna().apply(pd.Series)

        X = [pad(X_y.pre.values, pre),
            pad(X_y.word.values, word),
            pad(X_y.post.values, post)]
        
        y = X_y.label.values

        return X, y, X_y.index

    def get_embedding_weights(self, word_index , src_path = '/home/corpora/word_embeddings', embedding = 'glove.6B.100d.txt'):

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


    def get_class_weights(self, labels):
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

    def run(self, data, return_model = False):  

        hparams = self.hparams

        pre_input = tf.keras.Input(shape=(self.pre,), dtype="int64")
        word_input = tf.keras.Input(shape=(self.word,), dtype="int64")
        post_input = tf.keras.Input(shape=(self.post,), dtype="int64")
        
        pre_embedding = layers.Embedding(self.embedding_matrix.shape[0], 
                                100, 
                                weights=[self.embedding_matrix],
                                input_length = self.pre, trainable=True)

        word_embedding = layers.Embedding(self.embedding_matrix.shape[0], 
                            100, 
                            weights=[self.embedding_matrix],
                            input_length = self.word, trainable=True)

        post_embedding = layers.Embedding(self.embedding_matrix.shape[0], 
                            100, 
                            weights=[self.embedding_matrix],
                            input_length = self.post, trainable=True)
        
        pre_embedded = pre_embedding(pre_input)
        word_embedded = word_embedding(word_input)
        post_embedded = post_embedding(post_input)

        merged = tf.keras.layers.concatenate([pre_embedded, word_embedded, post_embedded], axis=1)
        input_length = self.pre + self.word + self.post

        layer =  layers.Bidirectional(layers.LSTM(input_length*hparams['model_scale']))(merged)

        model_scale = hparams['model_scale']

        for _ in range(hparams['n_layers']):
            layer = layers.Dense(input_length*hparams['model_scale'], activation=hparams['activation'])(layer)
            layer = tf.keras.layers.Dropout(hparams['dropout'])(layer)
            model_scale = model_scale / 2

        output = layers.Dense(1, activation='sigmoid')(layer)

        model = tf.keras.Model(
            inputs=[pre_input, word_input, post_input],
            outputs=[output],
        )

        opt = Adam(lr = hparams['lr'])

        model.compile(optimizer = opt, 
                    loss = 'binary_crossentropy', 
                    metrics = self.metrics)

        class_weight = self.get_class_weights(data['y_train'])

        model.fit(  data['X_train'] , 
                    data['y_train'],
                    batch_size=hparams['batch_size'],
                    validation_data=(data['X_val'], data['y_val']),
                    epochs=hparams['epochs'],
                    verbose = 1,
                    callbacks= self.callbacks,
                    class_weight = class_weight)

        if return_model:
            return model
        else:
            scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
            return scores


class SpanBert:

    def __init__(self, train, val, test, hparams):

        self.train = train
        self.val = val
        self.test = test
        self.hparams = hparams

    def get_data(self):
        
        X_train = bert_prep(train.text.to_list(), max_len = MAX_LEN)
        X_val = bert_prep(val.text.to_list(), max_len = MAX_LEN)
        X_test = bert_prep(test.text.to_list(), max_len = MAX_LEN)

        y_train = np.vstack(pad_sequences(train.word_mask.values, 
                                        maxlen = MAX_LEN, 
                                        truncating = 'post', 
                                        padding = 'post'))

        y_val = np.vstack(pad_sequences(val.word_mask.values, 
                                        maxlen = MAX_LEN, 
                                        truncating = 'post', 
                                        padding = 'post'))

        y_test = np.vstack(pad_sequences(test.word_mask.values, 
                                        maxlen = MAX_LEN, 
                                        truncating = 'post', 
                                        padding = 'post'))

        train_samples = {'X_train' : X_train, 
                         'y_train' : y_train, 
                         'X_val' : X_val, 
                         'y_val' : y_val, 
                         'X_test' : X_test, 
                         'y_test' : y_test}

        return train_samples

    def bert_prep(self, text, max_len=128):
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

    def run(self, data, input_length, output_length, hparams, callbacks, metrics, loss = 'categorical_crossentropy', embedding_matrix = 0, return_model = False, task_f1 = True): 

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


class MaskBert:

    def __init__(self, train, val, text, input_length):
        self.train = train
        self.val = val
        self.test = test
        self.input_length = input_length

        self.metrics = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            AUC(name='auc')
        ]

    def get_data(self):

        X_train_ids, X_train_tok_types, X_train_attn_mask = bert_prep(\
                        self.train.text.to_list(), max_len = MAX_LEN)
        X_val_ids, X_val_tok_types, X_val_attn_mask = bert_prep(\
                        self.val.text.to_list(), max_len = MAX_LEN)
        X_test_ids, X_test_tok_types, X_test_attn_mask = bert_prep(\
                        self.test.text.to_list(), max_len = MAX_LEN)

        X_train_mask = np.vstack(data.loc[train_index].x_word_mask.values)
        X_val_mask = np.vstack(data.loc[val_index].x_word_mask.values)
        X_test_mask = np.vstack(data.loc[test_index].x_word_mask.values)

        y_train = np.array(data.loc[train_index].label)
        y_val = np.array(data.loc[val_index].label)
        y_test = np.array(data.loc[test_index].label)

        train_samples = {
            'X_train' : [X_train_ids, X_train_tok_types, X_train_attn_mask, X_train_mask],
            'X_val' : [X_val_ids, X_val_tok_types, X_val_attn_mask, X_val_mask],
            'X_test' : [X_test_ids, X_test_tok_types, X_test_attn_mask, X_test_mask],
            'y_train' : y_train ,
            'y_val' : y_val ,
            'y_test' : y_test 
        }

        return train_samples

    def run(self, data, return_model = False): 

        input_length = self.input_length
        hparams = self.hparams

        ids = tf.keras.layers.Input((input_length,), dtype=tf.int32)
        tok_types = tf.keras.layers.Input((input_length,), dtype=tf.int32)
        attn_mask = tf.keras.layers.Input((input_length,), dtype=tf.int32)
        target_mask = tf.keras.layers.Input((input_length,), dtype=tf.int32)
        
        config = BertConfig() 
        config.output_hidden_states = False # Set to True to obtain hidden states
        
        bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
        
        embedded = bert_model(ids, attention_mask=attn_mask, token_type_ids=tok_types)[0]
        target_tensor = layers.Dense(input_length)(target_mask)
        stacked = layers.Concatenate([embedding, target_tensor], axis=0)

        model_scale = hparams['model_scale']

        layer =  layers.LSTM(input_length*model_scale)(stacked)

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
                    metrics = self.metrics)

        class_weight = get_class_weights(data['y_train'])

        model.fit(data['X_train'] , 
                data['y_train'],
                batch_size=hparams['batch_size'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=hparams['epochs'],
                verbose = 1,
                callbacks= self.callbacks,
                class_weight = class_weight)

        scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)

        if return_model:
            return model
        else:
            scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
            return scores

    def bert_prep(self, text, max_len=128):
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



''' functional approach '''
 


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


def target_bert(data, input_length, hparams, callbacks, metrics, embedding_matrix = 0): 

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

def bert_to_mask(data, input_length, output_length, hparams, callbacks, metrics, loss = 'categorical_crossentropy', embedding_matrix = 0, return_model = False, task_f1 = True): 

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