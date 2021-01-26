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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

import transformers
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification, BertConfig, BertTokenizerFast
from transformers import TFBertForTokenClassification
from transformers import TFElectraForTokenClassification, ElectraTokenizerFast
from transformers import RobertaTokenizerFast, TFRobertaForTokenClassification

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

class TokenBert:

    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen= maxlen
        self.name = method_name
        self.tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
        self.model = TFElectraForTokenClassification.from_pretrained('google/electra-small-discriminator', num_labels=3)
        
        self.add_bert_sequences()

        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        ]
        
        _labels = [0,1,2]
        self.labels = {k:v for k,v in zip(_labels, list(to_categorical(_labels)))}
        self.weights = {0:1, 1:1, 2:1}
    

    def bert_prep(self, tokens, max_len=128):
        
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)

        input_ids = []
        token_type_ids = []
        attn_mask = []
        word_ids = []

        for i in tqdm(range(len(tokens))):
            encoding = self.tokenizer(tokens[i], 
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

    def make_target_labels(self, row, field = 'word_mask'):
        
        labels = [row[field][word_idx] if word_idx != None else 2 for word_idx in row.word_ids]        
        sequences = np.vstack([self.labels[s] for s in labels]).T
        
        return sequences 

    def make_baseline_weights(self, row, field = 'baseline_word_mask'):
              
        sequences = [row[field][word_idx] if word_idx != None else -1 for word_idx in row.word_ids]
        sequences = np.array([s if s == 1 else 0.5 for s in sequences])
        # sequences = np.vstack([np.array(s) for s in [sequences] * 3])
        return sequences 
    
    def get_class_weights(self, sequences):

        try:
            self.weights = {
                0 : self.hparams['neg_weight'],
                1 : self.hparams['pos_weight'],
                2 : self.hparams['pad_weight'],
            }

        except KeyError:
            pass

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

        try:

            self.baseline_train  = np.vstack(self.train.apply(self.make_baseline_weights, 
                                    field = 'baseline_word_mask', axis=1).values)

            self.baseline_val  = np.vstack(self.val.apply(self.make_baseline_weights, 
                                    field = 'baseline_word_mask', axis=1).values)

            self.baseline_test  = np.vstack(self.test.apply(self.make_baseline_weights, 
                                    field = 'baseline_word_mask', axis=1).values)

        except KeyError:
            pass
        
        self.train_weights = self.get_class_weights(self.y_train).astype(float)

        return None

    def task_results(self, y_true, y_pred):
        df = pd.DataFrame()
        df['pred'] = list(y_pred)
        df['true'] = list(y_true)

        df['pred_mask'] = df.pred.apply(np.array).apply(lambda x : np.argmax(x,-1))
        df['true_mask'] = df.true.apply(np.array).apply(lambda x : np.argmax(x,-1))

        df['pred_word_level'] = df.pred_mask.apply(lambda x : np.where(x == 1)[0])
        df['true_word_level'] = df.true_mask.apply(lambda x : np.where(x == 1)[0])

        df['f1_score'] = df.apply(lambda row : f1(row.pred_word_level, row.true_word_level), axis = 1)
        df['recall'] = df.apply(lambda row : recall(row.pred_word_level, row.true_word_level), axis = 1)
        df['precision'] = df.apply(lambda row : precision(row.pred_word_level, row.true_word_level), axis = 1)

        self._preds = df
        
        print('\n\t Word Level Positive F1 : ', df.f1_score.mean())

        return df.f1_score.mean()

    def run(self, data, return_model = False): 
        
        hp = self.hparams
        
        ids = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        tok_types = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        attn_mask = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
    
        layer = self.model([ids,attn_mask,tok_types])[0]

        layer = layers.Dense(3)(layer)
        layer = layers.Dropout(hp['dropout'])(layer)
        layer = layers.Dense(3)(layer)
        # layer = layers.Dropout(hp['dropout'])(layer)
        # layer = layers.Dense(3)(layer)
    
        # for _ in range(hp['n_layers']):
        #     layer = layers.Dense(hp['nodes'])(layer)
        #     layer = layers.Dropout(hp['dropout'])(layer)

        out = layers.Dense(3, activation='softmax')(layer) 

        model = tf.keras.Model( inputs=[ids, 
                                        tok_types, 
                                        attn_mask,
                                        # class_weights
                                        ], 
                                outputs=out)

        

        # model.summary()
        # loss = tf.keras.losses.SparseCategoricalCrossentropy()
        opt = Adam(lr = hp['lr'])
        model.compile(optimizer = opt, 
                    loss = 'categorical_crossentropy', 
                    metrics = self.metrics)
                    
        # model.summary()

        model.fit(  self.X_train , 
                    self.y_train,
                    batch_size=hp['batch_size'],
                    validation_data=(self.X_val, self.y_val),
                    epochs=hp['epochs'],
                    verbose = 1,
                    callbacks= self.callbacks,
                    sample_weight = self.train_weights)

        self.y_pred = model.predict(self.X_test)
        self.task_score = self.task_results(self.y_test, self.y_pred)
        self.scores = model.evaluate(self.X_test, self.y_test, return_dict = True)
        self.scores['task_f1'] = self.task_score

        if return_model:
            return model
        else:
            return self.scores

    def evaluation_data(self, df):
        df['input_ids'], df['token_type_ids'], \
        df['attn_mask'], df['word_ids'] = \
        self.bert_prep(df.tokens.to_list(), max_len = self.maxlen)

        return [np.vstack(df.input_ids.values).astype(float),
                np.vstack(df.token_type_ids.values).astype(float),
                np.vstack(df.attn_mask.values).astype(float)]
        


class Roberta(TokenBert):

    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen= maxlen
        self.name = method_name
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
        self.bert_model = TFRobertaForTokenClassification.from_pretrained('roberta-base', num_labels=3)
        self.add_bert_sequences()

        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        ]
        
        _labels = [0,1,2]
        self.labels = {k:v for k,v in zip(_labels, list(to_categorical(_labels)))}
        self.weights = {0:.5, 1:2, 2:1}

    def bert_prep(self, tokens, max_len=128):
        
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)

        input_ids = []
        token_type_ids = []
        attn_mask = []
        word_ids = []

        for i in tqdm(range(len(tokens))):
            encoding = self.tokenizer(tokens[i], 
                                max_length = max_len, 
                                truncation=True,
                                is_split_into_words=True,
                                padding='max_length',
                                )
            
            input_ids.append(encoding["input_ids"])
            token_type_ids.append(None)
            attn_mask.append(encoding["attention_mask"])
            word_ids.append(encoding.word_ids())

        return input_ids, token_type_ids, attn_mask, word_ids

    def run(self, data, return_model = False): 
        
        hp = self.hparams
        
        ids = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        # tok_types = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        attn_mask = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
    
        layer = self.bert_model(input_ids = ids,
                                attention_mask = attn_mask,
                                # token_type_ids = tok_types
                                )[0]

        # lstm = layers.Bidirectional(layers.LSTM(self.maxlen))(layer)
        # weights = tf.expand_dims(tf.square(class_weights), axis = -1)
        # weights = tf.repeat(weights, repeats = 3, axis = 2)
        # layer = layers.Multiply()([layer, weights])

        layer = layers.Dense(6)(layer)
        layer = layers.Dropout(hp['dropout'])(layer)
        layer = layers.Dense(6)(layer)
        layer = layers.Dropout(hp['dropout'])(layer)
        layer = layers.Dense(3)(layer)

        out = layers.Dense(3, activation='softmax')(layer) 

        model = tf.keras.Model( inputs=[ids, 
                                        # tok_types, 
                                        attn_mask,
                                        # class_weights
                                        ], 
                                outputs=out)

        

        # model.summary()

        opt = Adam(lr = hp['lr'])
        model.compile(optimizer = opt, 
                    loss = 'categorical_crossentropy', 
                    metrics = self.metrics)
                    
        # model.summary()

        model.fit(  self.X_train , 
                    self.y_train,
                    batch_size=hp['batch_size'],
                    validation_data=(self.X_val, self.y_val),
                    epochs=hp['epochs'],
                    verbose = 1,
                    callbacks= self.callbacks,
                    sample_weight = self.train_weights)
                    
        self.y_pred = model.predict(self.X_test)
        self.task_score = self.task_results(self.y_test, self.y_pred)
        self.scores = model.evaluate(self.X_test, self.y_test, return_dict = True)
        self.scores['task_f1'] = self.task_score

        if return_model:
            return model
        else:
            return self.scores

from transformers import AlbertTokenizerFast, TFAlbertForTokenClassification

class Albert(TokenBert):


    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen= maxlen
        self.name = method_name
        self.tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
        self.model = TFAlbertForTokenClassification.from_pretrained('albert-base-v2')        
        self.add_bert_sequences()

        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        ]
        
        _labels = [0,1,2]
        self.labels = {k:v for k,v in zip(_labels, list(to_categorical(_labels)))}
        self.weights = {0:1, 1:1, 2:1}


    def run(self, data, return_model = False): 
        
        hp = self.hparams
        
        ids = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        attn_mask = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        # electra = tf.keras.layers.Input((self.maxlen, 3), dtype=tf.float32)
        # roberta = tf.keras.layers.Input((self.maxlen, 3), dtype=tf.float32)
    
        layer = self.model(input_ids = ids,
                           attention_mask = attn_mask
                          )[0]

        layer = tf.keras.layers.Dense(3)(layer)
        layer = tf.keras.layers.Dropout(hp['dropout'])(layer)
        layer = tf.keras.layers.Dense(3)(layer)
        layer = tf.keras.layers.Dropout(hp['dropout'])(layer)
        layer = tf.keras.layers.Dense(3)(layer)

        # electra_nn = tf.keras.layers.Dense(3)(electra)
        # roberta_nn = tf.keras.layers.Dense(3)(roberta)

        # layer = layer * electra_nn
        # layer = layer * roberta_nn
        
        out = tf.keras.layers.Dense(3, activation='softmax')(layer) 

        model = tf.keras.Model( inputs=[ids, 
                                        # tok_types, 
                                        attn_mask,
                                        # class_weights
                                        ], 
                                outputs=out)

        

        # model.summary()

        opt = tf.keras.optimizers.Adam(lr = hp['lr'])

        model.compile(optimizer = opt, 
                    loss = 'categorical_crossentropy', 
                    metrics = self.metrics)
                    
        # model.summary()

        model.fit(  self.X_train , 
                    self.y_train,
                    batch_size=hp['batch_size'],
                    validation_data=(self.X_val, self.y_val),
                    epochs=hp['epochs'],
                    verbose = 1,
                    callbacks= self.callbacks,
                    sample_weight = self.train_weights)
                    
        self.y_pred = model.predict(self.X_test)
        self.task_score = self.task_results(self.y_test, self.y_pred)
        self.scores = model.evaluate(self.X_test, self.y_test, return_dict = True)
        self.scores['task_f1'] = self.task_score

        if return_model:
            return model
        else:
            return self.scores

from transformers import AlbertTokenizerFast, TFAlbertForTokenClassification

class AlbertAndFriends(TokenBert):


    def __init__(self, train, val, test, method_name = '', maxlen = 128):

        self.train = train
        self.val = val
        self.test = test
        self.maxlen= maxlen
        self.tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
        self.model = TFAlbertForTokenClassification.from_pretrained('albert-base-v2')        
        
        self.methods = ['ELECTRA', 'ROBERTA', 'ALBERT']

        self.add_bert_sequences()
        self.get_other_predictions()

        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        ]
        
        _labels = [0,1,2]
        self.labels = {k:v for k,v in zip(_labels, list(to_categorical(_labels)))}
        self.weights = {0:1, 1:1, 2:1}

    def get_data(self):

        self.X_train = [np.vstack(self.train.input_ids.values).astype(float),
                        np.vstack(self.train.attn_mask.values).astype(float),
                        self.electra_train, self.roberta_train]

        self.X_val =   [np.vstack(self.val.input_ids.values).astype(float),
                        np.vstack(self.val.attn_mask.values).astype(float),
                        self.electra_val, self.roberta_val]

        self.X_test  = [np.vstack(self.test.input_ids.values).astype(float),
                        np.vstack(self.test.attn_mask.values).astype(float),
                        self.electra_test, self.roberta_test]

        self.y_train  = np.dstack(self.train.apply(self.make_target_labels, axis=1).values).T
        self.y_val  = np.dstack(self.val.apply(self.make_target_labels, axis=1).values).T
        self.y_test  = np.dstack(self.test.apply(self.make_target_labels, axis=1).values).T
        
        self.train_weights = self.get_class_weights(self.y_train).astype(float)

        return None

    def get_best_results(self, pred_dir):

        prediction_files = os.listdir(pred_dir)

        bestest = [[[float(f.strip('%s_' % m).strip('.json')), f] \
                    for f in prediction_files if m in f]\
                    .sort(key=lambda x: x[0])[0][1] for m in self.methods]

        df = pd.concat([pd.read_json(os.path.join(pred_dir, filename))\
            .add_prefix('%s_' % filename.split(_)[0])\
            for filename in bestest
            ], axis = 1)

        return df

    def get_other_predictions(self):

        fold_dir = [str(n) for n in range(5)]
        self.other_models = pd.DataFrame()

        for fold in fold_dir:
            data_dir = os.path.join('/home/burtenshaw/now/spans_toxic/data', fold)
            pred_dir = os.path.join('/home/burtenshaw/now/spans_toxic/predictions', fold)

            test = pd.read_pickle(os.path.join(data_dir, 'test.bin'))

            df = self.get_best_results(pred_dir)
            
            df.index = test.index

            self.other_models = pd.concat([self.other_models, df[['electra_y_pred', 'roberta_y_pred']]],
                                        ignore_index = False, axis= 0)
        
        self.electra_train = np.dstack(self.other_models.loc[self.train.index]\
                            .electra_y_pred.values).reshape(\
                            self.train.index.shape[0], self.maxlen, 3)
        self.electra_val = np.dstack(self.other_models.loc[self.val.index]\
                            .electra_y_pred.values).reshape(\
                            self.val.index.shape[0], self.maxlen, 3)
        self.electra_test = np.dstack(self.other_models.loc[self.test.index]\
                            .electra_y_pred.values).reshape(\
                            self.test.index.shape[0], self.maxlen, 3)

        self.roberta_train = np.dstack(self.other_models.loc[self.train.index]\
                            .roberta_y_pred.values).reshape(\
                            self.train.index.shape[0], self.maxlen, 3)
        self.roberta_val = np.dstack(self.other_models.loc[self.val.index]\
                            .roberta_y_pred.values).reshape(\
                            self.val.index.shape[0], self.maxlen, 3)
        self.roberta_test = np.dstack(self.other_models.loc[self.test.index]\
                            .roberta_y_pred.values).reshape(\
                            self.test.index.shape[0], self.maxlen, 3)

    def lr_scheduler(self, epoch, lr):
        return lr * self.hparams['epsilon']

    def run(self, data, return_model = False): 
        
        hp = self.hparams
        
        ids = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        attn_mask = tf.keras.layers.Input((self.maxlen,), dtype=tf.int32)
        electra = tf.keras.layers.Input((self.maxlen, 3), dtype=tf.float32)
        roberta = tf.keras.layers.Input((self.maxlen, 3), dtype=tf.float32)
    
        layer = self.model(input_ids = ids,
                           attention_mask = attn_mask
                          )[0]

        layer = tf.keras.layers.Dense(hp['nodes'])(layer)

        electra_nn = tf.keras.layers.Dense(hp['nodes'])(electra)
        electra_nn = tf.keras.layers.Dropout(hp['dropout'])(electra_nn)
        roberta_nn = tf.keras.layers.Dense(hp['nodes'])(roberta)
        roberta_nn = tf.keras.layers.Dropout(hp['dropout'])(roberta_nn)

        layer = tf.keras.layers.Multiply()([layer, electra_nn])
        layer = tf.keras.layers.Dense(hp['nodes'])(layer)
        layer = tf.keras.layers.Multiply()([layer, roberta_nn])
        layer = tf.keras.layers.Dense(hp['nodes'])(layer)

        
        out = tf.keras.layers.Dense(3, activation='softmax')(layer) 

        model = tf.keras.Model( inputs=[ids,
                                        attn_mask,
                                        electra,
                                        roberta
                                        ], 
                                outputs=out)

        
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = hp['lr']), 
                    loss = 'categorical_crossentropy', 
                    metrics = self.metrics)
                    
        self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1))
        self.X_train.append(self.electra_train)
        self.X_val.append(self.electra_val)
        self.X_test.append(self.electra_test)

        self.X_train.append(self.roberta_train)
        self.X_val.append(self.roberta_val)
        self.X_test.append(self.roberta_test)

        model.fit(  self.X_train , 
                    self.y_train,
                    batch_size=hp['batch_size'],
                    validation_data=(self.X_val, self.y_val),
                    epochs=hp['epochs'],
                    verbose = 1,
                    callbacks= self.callbacks,
                    sample_weight = self.train_weights)
                    
        self.y_pred = model.predict(self.X_test)
        task_score = self.task_results(self.y_test, self.y_pred)

        if return_model:
            return model
        else:
            scores = model.evaluate(self.X_test, self.y_test, return_dict = True)
            scores['task_f1'] = task_score
            return scores


def to_ensemble(y_pred, method, output_dir):
    filename = '%s_%s.json' % (method.name, str(method.task_score))
    save_path = os.path.join(output_dir, filename)
    df = pd.DataFrame()
    df['word_ids'] = method.test.word_ids
    df['y_pred'] = list(y_pred)
    df['labels'] = list(method.y_test)
    df.to_json(save_path)

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


