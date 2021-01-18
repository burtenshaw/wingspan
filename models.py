import pandas as pd
import numpy as np
import os
import io
from tqdm import *
import tensorflow as tf
# import tensorflow_hub as hub

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
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification, BertConfig, BertTokenizerFast
from transformers import TFBertForTokenClassification
import transformers

from sklearn.model_selection import train_test_split


class SiameseBert:

    def __init__(self, train, val, test, method_name = '', maxlen = 200):

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

    def __init__(self, train, val, test, method_name, max_len = 128):

        self.train = train
        self.val = val
        self.test = test
        self.method_name = method_name
        self.max_len = max_len
        self.output_length = max(train[method_name].max(), 
                                 val[method_name].max(), 
                                 test[method_name].max()) + 1
        self.metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')]

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

        self.X_train = self.bert_prep(train.text.to_list(), max_len = self.max_len)
        self.y_train = to_categorical(train[self.method_name].values, num_classes = self.output_len)
        self.X_val = self.bert_prep(val.text.to_list(), max_len = self.max_len)
        self.y_val = to_categorical(val[self.method_name].values, num_classes = self.output_len)
        self.X_test = self.bert_prep(test.text.to_list(), max_len = self.max_len)
        self.y_test = to_categorical(test[self.method_name].values, num_classes = self.output_len)

        return None

    def run(self, data, verbose = 1, return_model = False): 

        hp = self.hparams

        ids_input = layers.Input((self.max_len,), dtype=tf.int32)
        token_type_input = layers.Input((self.max_len,), dtype=tf.int32)
        attn_mask_input = layers.Input((self.max_len,), dtype=tf.int32)
        
        config = BertConfig() 
        config.output_hidden_states = False # Set to True to obtain hidden states
        
        bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
        
        embedded = bert_model(ids_input, attention_mask=attn_mask_input, token_type_ids=token_type_input)[0]
        
        model_scale = hp['model_scale']

        layer =  layers.Dense(self.max_len*model_scale)(embedded)
        layer = layers.Dense(self.output_length)(layer)
        layer = tf.keras.layers.GlobalAveragePooling1D()(layer)

        out = layers.Activation(K.activations.softmax)(layer)

        model = tf.keras.Model(inputs=[ids_input, 
                                    token_type_input, 
                                    attn_mask_input], 
                            outputs=out)

        # loss = K.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer = Adam(lr = hp['lr']), 
                    loss = 'categorical_crossentropy', 
                    metrics = metrics)

        model.fit(data['X_train'], 
                data['y_train'],
                batch_size=hp['batch_size'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=hp['epochs'],
                verbose = verbose,
                callbacks= callbacks)

        if return_model:
            return model
        else:
            scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
            return scores


class SiameseLSTM:

    def __init__(self, train, val, test, method_name = '', ):

        self.train = train
        self.val = val
        self.test = test

        self.make_vocab()

        self.metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
        ]

    def get_data(self):

        self.pre = self.hparams['pre']
        self.word = self.hparams['word']
        self.post = self.hparams['post']

        X_train, y_train, self.train_index = make_context_data(self.train, 
                                                pre = self.pre, 
                                                post = self.post, 
                                                word = self.word)
        
        X_val, y_val, self.val_index = make_context_data(self.val, 
                                            pre = self.pre, 
                                            post = self.post, 
                                            word = self.word)
        
        X_test, y_test, self.test_index = make_context_data(self.test, 
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

    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen = maxlen

        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy'),
        ]



    def get_data(self):
        
        X_train = self.bert_prep(self.train.text.to_list(), max_len = self.maxlen)
        X_val = self.bert_prep(self.val.text.to_list(), max_len = self.maxlen)
        X_test = self.bert_prep(self.test.text.to_list(), max_len = self.maxlen)

        y_train = np.vstack(pad_sequences(self.train.word_mask.values, 
                                        maxlen = self.maxlen, 
                                        truncating = 'post', 
                                        padding = 'post'))

        y_val = np.vstack(pad_sequences(self.val.word_mask.values, 
                                        maxlen = self.maxlen, 
                                        truncating = 'post', 
                                        padding = 'post'))

        y_test = np.vstack(pad_sequences(self.test.word_mask.values, 
                                        maxlen = self.maxlen, 
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

    def run(self, data, return_model = False): 

        input_length = self.maxlen 
        output_length = self.maxlen
        hparams = self.hparams

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
                    loss = 'categorical_crossentropy', 
                    metrics = self.metrics)

        model.fit(data['X_train'] , 
                data['y_train'],
                batch_size=hparams['batch_size'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=hparams['epochs'],
                verbose = 1,
                callbacks= self.callbacks)

        if return_model:
            return model
        else:
            scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)
            
            return scores


class MaskBert:

    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.input_length = self.maxlen = maxlen

        self.add_bert_sequences()

        self.train = self.make_target_mask(train)
        self.val = self.make_target_mask(val)
        self.test = self.make_target_mask(test)

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

        ready = lambda x : np.vstack(x).astype('float32')

        return {
            'X_train' : [ready(self.train.input_ids), 
                         ready(self.train.token_type_ids), 
                         ready(self.train.attn_mask), 
                         ready(self.train.target_mask.values)],

            'X_val' : [ready(self.val.input_ids), 
                       ready(self.val.token_type_ids), 
                       ready(self.val.attn_mask), 
                       ready(self.val.target_mask.values)],

            'X_test' : [ready(self.test.input_ids.values), 
                        ready(self.test.token_type_ids.values), 
                        ready(self.test.attn_mask.values), 
                        ready(self.test.target_mask.values)],

            'y_train' : np.array(self.train.word_mask).astype('float32'),
            'y_val' : np.array(self.val.word_mask).astype('float32'),
            'y_test' : np.array(self.test.word_mask).astype('float32') 
        }
         
    def pad_mask(self,mask):
        mask = np.array(mask[:self.maxlen])
        return np.pad(mask, (0,self.maxlen - mask.shape[0]))

    def make_target_mask(self, data):
        data = data.explode('word_mask')
        data['target_mask'] = data.groupby(level=0).apply(len)\
            .apply(lambda x : [to_categorical(y) for y in range(x)])\
                .explode().apply(self.pad_mask).values
        return data

    def add_bert_sequences(self):
        
        self.train['input_ids'], self.train['token_type_ids'], self.train['attn_mask'] = \
            [x.tolist() for x in bert_prep(self.train.text.to_list(), max_len = self.maxlen)]
        self.val['input_ids'], self.val['token_type_ids'], self.val['attn_mask'] = \
            [x.tolist() for x in bert_prep(self.val.text.to_list(), max_len = self.maxlen)]
        self.test['input_ids'], self.test['token_type_ids'], self.test['attn_mask'] = \
            [x.tolist() for x in bert_prep(self.test.text.to_list(), max_len = self.maxlen)]

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
        target_tensor = layers.LeakyReLU()(target_tensor)
        target_tensor = tf.expand_dims(target_tensor, -1)

        stacked = layers.concatenate([embedded, target_tensor], axis=2)
        # stacked = tf.stack([embedded, target_tensor], axis=2)

        print(stacked.shape)
        layer =  layers.LSTM(input_length)(stacked)
        print(layer.shape)
        layer = layers.Dense(input_length, activation=hparams['activation'])(layer)
        # for _ in range(hparams['n_layers']):
        #     layer = layers.Dense(input_length, activation=hparams['activation'])(layer)
        #     layer = tf.keras.layers.Dropout(hparams['dropout'])(layer)
        print(layer.shape)
        layer = tf.multiply(layer, target_tensor)
        print(target_tensor.shape)
        
        print(layer.shape)
        layer = layers.GlobalAveragePooling1D()(layer)
        print(layer.shape)
        out = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
        model = tf.keras.Model(inputs=[ids, 
                                       tok_types, 
                                       attn_mask,
                                       target_mask], 
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

        if return_model:
            return model
        else:
            return model.evaluate(data['X_test'], data['y_test'], return_dict = True)

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

class BinaryTruePositives(tf.keras.metrics.Metric):

  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives

  def reset_states(self):
    self.true_positives.assign(0)

class TokenBert:

    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen= maxlen
        self.add_bert_sequences()

        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
        ]
        
        _labels = [0,1,2]
        self.labels = {k:v for k,v in zip(_labels, list(to_categorical(_labels)))}
        self.weights = {0:.5, 1:5, 2:0}

    def bert_prep(self, tokens, max_len=128):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        input_ids = []
        token_type_ids = []
        attn_mask = []
        word_ids = []

        for i in tqdm(range(len(tokens))):
            encoding = tokenizer(tokens[i], 
                                max_length = max_len, 
                                truncation=True,
                                is_split_into_words=True,
                                padding='max_length',
                                )
            
            input_ids.append(encoding["input_ids"])
            token_type_ids.append(encoding['token_type_ids'])
            attn_mask.append(encoding["attention_mask"])
            word_ids.append(encoding.word_ids())
            
        return input_ids, token_type_ids, attn_mask, word_ids

    def make_target_labels(self, row):
        
        labels = [row.word_mask[word_idx] if word_idx != None else 2 for word_idx in row.word_ids]        
        sequences = np.vstack([self.labels[s] for s in labels]).T
        
        return sequences 
    
    def get_class_weights(self, sequences):
        return np.vectorize(lambda x : self.weights[x])(np.argmax(sequences, axis =2))
        
    def add_bert_sequences(self):

        self.train['input_ids'], self.train['token_type_ids'], \
        self.train['attn_mask'], self.train['word_ids'] = \
        self.bert_prep(self.train.tokens.to_list(), max_len = self.maxlen)

        self.val['input_ids'], self.val['token_type_ids'], \
        self.val['attn_mask'], self.val['word_ids'] = \
        self.bert_prep(self.val.tokens.to_list(), max_len = self.maxlen)

        self.test['input_ids'], self.test['token_type_ids'], \
        self.test['attn_mask'], self.test['word_ids'] = \
        self.bert_prep(self.test.tokens.to_list(), max_len = self.maxlen)

    def get_data(self):

        self.X_train = [np.vstack(self.train.input_ids.values).astype(float),
                        np.vstack(self.train.token_type_ids.values).astype(float),
                        np.vstack(self.train.attn_mask.values).astype(float)]
        # print(self.X_train.shape)
        self.X_val =   [np.vstack(self.val.input_ids.values).astype(float),
                        np.vstack(self.val.token_type_ids.values).astype(float),
                        np.vstack(self.val.attn_mask.values).astype(float)]
        # print(self.X_val.shape)
        self.X_test  = [np.vstack(self.test.input_ids.values).astype(float),
                        np.vstack(self.test.token_type_ids.values).astype(float),
                        np.vstack(self.test.attn_mask.values).astype(float)]
        # print(self.X_test.shape)

        self.y_train  = np.dstack(self.train.apply(self.make_target_labels, axis=1).values).T
        self.y_val  = np.dstack(self.val.apply(self.make_target_labels, axis=1).values).T
        self.y_test  = np.dstack(self.test.apply(self.make_target_labels, axis=1).values).T

        # self.X_train.append(self.get_class_weights(self.y_train).astype(float))
        # self.X_val.append(self.get_class_weights(self.y_val).astype(float))
        # self.X_test.append(self.get_class_weights(self.y_test).astype(float))

        return None


    def task_metrics(self, true, pred):
        
        true = np.argmax(true[1:-1,:2],-1)
        true = list(np.where(true == 1)[0])
        pred = np.argmax(pred[1:-1,:2],-1)
        pred = list(np.where(pred == 1)[0]) 
        
        p = precision(pred, true)
        r = recall(pred, true)
        f = f1(pred, true)
        
        return p,r,f

    def task_results(self, y_true, y_pred):
        df = pd.DataFrame()
        df['pred'] = list(y_pred)
        df['true'] = list(y_true)
        out = df.apply(lambda x : \
                self.task_metrics(x.true, x.pred), axis = 1, \
                    result_type='expand')
        out.columns = ['precision','recall','f1']
        print(out.mean())

    # def accuracy(self, true, pred):
    #     true = tf.argmax(true,2)
    #     pred = tf.argmax(pred,2)
    #     m = tf.keras.metrics.SensitivityAtSpecificity(0.5)
    #     m.update_state(true,)
    #     return tf.keras.metrics.categorical_accuracy(true, pred)

    def run(self, data, return_model = False): 
        
        hp = self.hparams
        
        ids = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        tok_types = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        attn_mask = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        # class_weights = tf.keras.layers.Input((self.maxlen,), dtype=tf.float32)
        
        bert_model = TFBertForTokenClassification.from_pretrained('bert-base-cased', num_labels=3)
        layer = bert_model([ids,attn_mask,tok_types])[0]

        # lstm = layers.Bidirectional(layers.LSTM(self.maxlen))(layer)
        # weights = tf.expand_dims(tf.square(class_weights), axis = -1)
        # weights = tf.repeat(weights, repeats = 3, axis = 2)
        # layer = layers.Multiply()([layer, weights])
        out = layers.Dense(3, activation='softmax')(layer) 

        model = tf.keras.Model( inputs=[ids, 
                                        tok_types, 
                                        attn_mask], 
                                outputs=out)

        model.summary()
        # loss = tf.keras.losses.SparseCategoricalCrossentropy()
        opt = Adam(lr = hp['lr'])
        model.compile(optimizer = opt, 
                    loss = 'categorical_crossentropy', 
                    metrics = self.metrics + [tf.keras.metrics.SensitivityAtSpecificity(.5)])
                    
        model.summary()
        print(len(self.X_train))
        model.fit(  self.X_train , 
                    self.y_train,
                    batch_size=hp['batch_size'],
                    validation_data=(self.X_val, self.y_val),
                    epochs=hp['epochs'],
                    verbose = 1,
                    callbacks= self.callbacks)

        y_pred = model.predict(self.X_test)
        self.task_results(self.y_test, y_pred)

        if return_model:
            return model
        else:
            scores = model.evaluate(self.X_test, self.y_test, return_dict = True)
            return scores


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