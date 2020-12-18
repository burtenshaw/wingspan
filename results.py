import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_curve

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


#!/usr/bin/env python
import sys
import os
import os.path
from scipy.stats import sem
import numpy as np
from ast import literal_eval

def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)


def evaluate(pred_lines, gold_lines):
    """
    Based on https://github.com/felipebravom/EmoInt/blob/master/codalab/scoring_program/evaluation.py
    :param pred: file with predictions
    :param gold: file with ground truth
    :return:
    """
    # # read the predictions
    # pred_lines = pred.readlines()
    # # read the ground truth
    # gold_lines = gold.readlines()

    # only when the same number of lines exists
    if (len(pred_lines) == len(gold_lines)):
        data_dic = {}
        for n, line in enumerate(gold_lines):
            parts = line.split('\t')
            if len(parts) == 2:
                data_dic[int(parts[0])] = [literal_eval(parts[1])]
            else:
                raise ValueError('Format problem for gold line %d.', n)

        for n, line in enumerate(pred_lines):
            parts = line.split('\t')
            if len(parts) == 2:
                if int(parts[0]) in data_dic:
                    try:
                        data_dic[int(parts[0])].append(literal_eval(parts[1]))
                    except ValueError:
                        # Invalid predictions are replaced by a default value
                        data_dic[int(parts[0])].append([])
                else:
                    raise ValueError('Invalid text id for pred line %d.', n)
            else:
                raise ValueError('Format problem for pred line %d.', n)

        # lists storing gold and prediction scores
        scores = []
        for id in data_dic:
            if len(data_dic[id]) == 2:
                gold_spans = data_dic[id][0]
                pred_spans = data_dic[id][1]
                scores.append(f1(pred_spans, gold_spans))
            else:
                sys.exit('Repeated id in test data.')

        return (np.mean(scores), sem(scores))

    else:
        print('Predictions and gold data have different number of lines.')




# class EvalResults:

#     def __init__(self, results, y_test, text = None, roc=False, bin_t=0.5, do_labels=False, token_type='char'):

#         rdf = pd.concat([pd.DataFrame(results),pd.DataFrame(y_test)], \
#                     keys=['pred', 'true'], axis=1)
        
#         self.probsdf = rdf.copy(deep=True)

#         if roc:
#             self.rocdf = pd.DataFrame(rdf.T.unstack().T.swaplevel(0,1).groupby(level=0)\
#                             .apply(lambda x : roc_curve(x.true, x.pred)).to_list(), \
#                                 columns= ['fpr', 'tpr', 'threshold'])

#             roc_dict = self.rocdf.apply(lambda row : row.threshold[np.argmax(row.tpr) - \
#                                                         np.argmax(row.fpr)], axis=1 ).to_dict()
#             rdf.pred = rdf.pred.apply(lambda col : np.where(col > roc_dict[col.name], 1, 0 ))
#         elif bin_t:
#             rdf.pred = rdf.pred.apply(lambda col : np.where(col > bin_t, 1, 0 ))
        
#         if token_type == 'char':
#             rdf['pred'] = rdf.pred.apply(character_mask_to_entities, axis=1, result_type='expand')
#             rdf['true'] = rdf.true.apply(character_mask_to_entities, axis=1, result_type='expand')
#         elif token_type == 'word':
#             rdf['text'] = text
#             rdf['pred'] = rdf.apply(word_mask_to_character_entity, field = 'pred', result_shape=results.shape[1], axis=1, result_type='expand')
#             rdf['true'] = rdf.apply(word_mask_to_character_entity, field = 'true', result_shape=results.shape[1], axis=1, result_type='expand')
        
#         self.rdf = rdf
#         self.evdf = pd.DataFrame()

#         self.evdf['f1'] = rdf.apply(lambda row : \
#             f1(row.pred[row.pred != 0 ], row.true[row.true != 0]), axis=1)
#         self.evdf['precision'] = rdf.apply(lambda row : \
#             precision(row.pred[row.pred != 0 ], row.true[row.true != 0]), axis=1)
#         self.evdf['recall'] = rdf.apply(lambda row : \
#             recall(row.pred[row.pred != 0 ], row.true[row.true != 0]), axis=1)
#         self.mean = self.evdf.mean()

#         if do_labels:
#             self.label_scores()
        
    
#     def label_scores(self):
        
#         metrics = ['precision','recall','F1','support']
#         prdf = pd.DataFrame(self.rdf.apply(lambda x : precision_recall_fscore_support(x.true, x.pred), axis=1)\
#                 .apply(lambda x : {m : {n:v for n, v in enumerate(s)} for m,s in zip(metrics, x)})\
#                 .to_list()).stack().apply(pd.Series)

#         prdf['metric'] = prdf.index.get_level_values(1)
#         prdf = prdf.groupby('metric').agg(np.mean).T

#         avdf = pd.concat([self.rdf.apply(lambda x : \
#                     precision_recall_fscore_support(x.true, x.pred, average=av),\
#                     axis=1, result_type='expand').mean().to_frame().T.rename(index={0:av}) \
#                     for av in ['macro', 'micro', 'weighted']])
        
#         avdf.columns= metrics
        
#         self.label_scores = pd.concat([prdf,avdf], names=[0,1]).drop(columns='support')
#         self.avdf = avdf
#         self.prdf = prdf