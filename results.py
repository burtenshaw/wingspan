import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_curve
import tensorflow as tf
from utils import *


class EvalResults:

    def __init__(self, component_models, results_df, params = {}, format_conversions = [], metrics = [], roc = False):
        test_index = results_df.index.drop_duplicates()

        _data = pd.read_pickle("data/train.bin").loc[test_index]

        format_conversions.extend([('entity', word_mask_to_character_entity), 
                    ('word_mask', lambda x, field, pad, result_shape : np.where(np.array(x[field]) == 1)[0])])

        metrics.extend([('f1', f1), 
                    ('precision', precision), 
                    ('recall' ,recall), 
                    ('span_mean', lambda x, y : np.mean(y)) 
                    ])
        
        edf = pd.DataFrame(index = test_index)
        edf['text'] = _data.text[test_index]
        edf['word_mask'] = _data.word_mask.apply(lambda x : np.where(np.array(x) == 1)[0])[test_index]
        edf['entity'] = _data.entities[test_index]

        colidx = pd.MultiIndex.from_product([[c for c,_ in format_conversions], [m for m,_ in metrics]])
        rdf = pd.DataFrame(columns=colidx)

        if roc:
            self.do_t = self.do_roc
            
        else:
            self.do_t = lambda x , t: np.where(x > t , 1 , 0)
            
        for model,t in component_models:
            prediction_label = '%s_pred' % model
            predictions = results_df[model].groupby(level=0).pred.apply(np.array).apply(self.do_t, t = t)
            edf[prediction_label] = predictions
            print(model)
            
            for format_label, format_func in format_conversions:
                model_format_label = '%s_%s' % (prediction_label, format_label)
                formatted_predictions = edf.apply(format_func, field = prediction_label, pad = False, result_shape=200, axis=1)
                edf[model_format_label] = formatted_predictions
                print('\t %s' % (format_label))

                for metric_label, metric in metrics:
                    model_metric_label = '%s_%s' % (model_format_label, metric_label)
                    metric_result = edf.apply(lambda row : metric(row[format_label], row[model_format_label]), axis = 1)
                    edf[model_metric_label] = metric_result
                    print('\t \t: %s : %s' % \
                        (metric_label, metric_result.mean()))
                    
                    # rdf.at[model, (format_label, metric_label)] = metric_result.mean()

            # rdf.at[model, 'support'] = edf[format_label].apply(len).sum()

            if model in params:
                for k, v in params[model].items():
                    rdf.at[model, ('params', k)] = v

        self.rdf = rdf
        self.edf = edf

    def do_roc(self):     

        pass 


        # self.rocdf = pd.DataFrame(rdf.T.unstack().T.swaplevel(0,1).groupby(level=0)\
        #                 .apply(lambda x : roc_curve(x.true, x.pred)).to_list(), \
        #                     columns= ['fpr', 'tpr', 'threshold'])

        # self.roc_dict = self.rocdf.apply(lambda row : row.threshold[np.argmax(row.tpr) - \
        #                                             np.argmax(row.fpr)], axis=1 ).to_dict()


def train_accuracy(y_true, y_pred):

    y_true = tf.make_ndarray(tf.make_tensor_proto(y_true))
    y_pred = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    
    y_pred = np.where(y_pred > 0.5, 1,0 )
    y_true = np.where(y_true > 0.5, 1,0 )

    score = f1(list(y_pred), list(y_true))

    return score