from tensorboard.plugins.hparams import api as hp
import models

METHODS = {
        'BERT_NGRAM' : models.SiameseBert,
        'BERT_SPAN' : models.SpanBert,
        'BERT_MASK' : models.MaskBert,
        'BERT_TOKEN' : models.TokenBert,
        'LSTM_NGRAM' : models.SiameseLSTM,
        'N_SPANS' : models.CategoricalBert,
        'LEN' : models.CategoricalBert,
        'START' : models.CategoricalBert,
        'END' : models.CategoricalBert
}

TUNING = {

        'CAT_N_SPANS' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([2,4]))
                ],
                
        'CAT_LEN' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([2,4])),
                ],

        'CAT_START' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([2,4])),
                ],

        'CAT_END' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16])),
                hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2])),
                hp.HParam('epochs', hp.Discrete([2,4])),
                ],

        'BERT_SPAN' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([8,16])),
                hp.HParam('lr', hp.Discrete([1e-3, 1e-5, 1e-7])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('n_layers', hp.Discrete([1])),
                hp.HParam('model_scale',hp.Discrete([1,2,3,4])),
                hp.HParam('epochs', hp.Discrete([3,4,5]))
                ],

        'BERT_TOKEN' : [
                hp.HParam('batch_size', hp.Discrete([16])),
                hp.HParam('lr', hp.Discrete([1e-4, 1e-5, 1e-6, 1e-7])),
                hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
                hp.HParam('epochs', hp.Discrete([3,4,5])),
                hp.HParam('neg_weight', hp.RealInterval(0.1,1.0)),
                hp.HParam('pos_weight', hp.RealInterval(5.0,15.0))
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

        'BERT_MASK' : [
                hp.HParam('activation', hp.Discrete(['relu'])),
                hp.HParam('batch_size', hp.Discrete([16])),
                hp.HParam('lr', hp.Discrete([1e-1,1e-3,1e-5,1e-7])),
                hp.HParam('dropout',hp.Discrete([0.2])),
                hp.HParam('n_layers', hp.Discrete([1,2])),
                hp.HParam('model_scale',hp.Discrete([1,2,3])),
                hp.HParam('epochs', hp.Discrete([3,4,5]))
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

        'BERT_SPAN' : { # optimised 7/01/21
                'activation' : 'relu',
                'batch_size' : 8,
                'lr' : 0.00002,
                'dropout' : 0.24,
                'n_layers' : 1,
                'model_scale' : 1, 
                'epochs' : 2
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

        'LSTM_NGRAM' : { # optimised 7/01/21
                'activation' : 'relu',
                'batch_size' : 256,
                'lr' : 0.0050000, 
                'dropout' : 0.23,
                'n_layers' : 2,
                'model_scale' : 2,
                'pre' : 25,
                'post' : 25,
                'word' : 1,
                'epochs' : 6 # could be optimised
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


