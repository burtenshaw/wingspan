from tensorboard.plugins.hparams import api as hp
import models

METHODS = {
        'BERT_NGRAM' : models.SiameseBert,
        'BERT_SPAN' : models.SpanBert,
        'BERT_MASK' : models.MaskBert,
        'LSTM_NGRAM' : models.SiameseLSTM,
        'CAT_N_SPANS' : models.CategoricalBert,
        'CAT_LEN' : models.CategoricalBert,
        'CAT_START' : models.CategoricalBert,
        'CAT_END' : models.CategoricalBert
}

TUNING = {

        'CAT_N_SPANS' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16,32])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([10])),
                ],
                
        'CAT_LEN' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16,32])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([10])),
                ],

        'CAT_START' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16,32])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([10])),
                ],

        'CAT_END' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16,32])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([10])),
                ],

        'BERT_SPAN' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1])),
                hp.HParam('model_scale',hp.Discrete([1,2,3])),
                hp.HParam('epochs', hp.Discrete([2]))
                ],

        'BERT_NGRAM' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([16])),
                hp.HParam('lr', hp.Discrete([1e-3,1e-5,1e-7])),
                hp.HParam('dropout',hp.Discrete([0.2])),
                hp.HParam('n_layers', hp.Discrete([2,3])),
                hp.HParam('model_scale',hp.Discrete([2,3])),
                hp.HParam('pre', hp.Discrete([64])),
                hp.HParam('post', hp.Discrete([64])),
                hp.HParam('word', hp.Discrete([0])),
                hp.HParam('epochs', hp.Discrete([2])),
                hp.HParam('lstm', hp.Discrete([True]))
                    ],

        'BERT_NGRAM' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([16])),
                hp.HParam('lr', hp.Discrete([1e-1,1e-3,1e-5,1e-7])),
                hp.HParam('dropout',hp.Discrete([0.2])),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2,3])),
                    ],

        'LSTM_NGRAM' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([32,64,128,256])),
                hp.HParam('lr', hp.Discrete([0.001, 0.005, 0.01,])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2,3])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('pre', hp.Discrete(range(10, 60))),
                hp.HParam('post', hp.Discrete(range(10, 60))),
                hp.HParam('word', hp.Discrete([1])),
                hp.HParam('epochs', hp.Discrete([6]))
                ],

        'ENSEMBLE' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16,32])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([10])),
                ]
}

BEST = {

        'CATEGORICAL' : {
                'activation': 'relu',
                'batch_size': 32,
                'lr': 7e-7,
                'dropout' : 0.115,
                'n_layers' : 1,
                'model_scale' : 1,
                'epochs' : 10,
        },

        'BERT_SPAN' : {
                'activation' : 'relu',
                'batch_size' : 8,
                'lr' : 7e-7, # to be optimised 
                'dropout' : 0.34,
                'n_layers' : 1,
                'model_scale' : 2, # could go up
                'epochs' : 3 # could go up
        },

        'BERT_NGRAM' : { # optimised 6/01/21
                'activation' : 'relu',
                'batch_size' : 8,
                'lr' : 1e-7,
                'dropout' : 0.2,
                'n_layers' : 1,
                'model_scale' : 1,
                'pre' : 64,
                'post' : 64,
                'word' : 0,
                'epochs' : 2,
                'lstm' : True
        },

        'LSTM_NGRAM' : {
                'activation' : 'relu',
                'batch_size' : 120,
                'lr' : 0.001, # could go down
                'dropout' : 0.23,
                'n_layers' : 4,
                'model_scale' : 4,
                'pre' : 15,
                'post' : 15,
                'word' : 1,
                'epochs' : 6 # could go up
        },

        'ENSEMBLE' : { # unkown as yet
                'activation' : 'relu',
                'batch_size' : 8,
                'lr' : 2e-2,
                'dropout' : .1,
                'n_layers' : 1,
                'model_scale' : 1,
                'epochs' : 10
        }
}


