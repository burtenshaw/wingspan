import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_curve

from utils import *



class EvalResults:

    def __init__(self, component_models, results_df, format_conversions = [], metrics = [], roc = False):
        test_index = results_df.index.drop_duplicates()

        _data = pd.read_pickle("data/train.bin").loc[test_index]

        format_conversions.extend([('entity', word_mask_to_character_entity), 
                    ('word_mask', lambda x, field, pad, result_shape : np.where(np.array(x[field]) == 1)[0])])

        metrics.extend([('f1', precision), 
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
            edf['%s_pred' % model] = results_df[model].groupby(level=0).pred.apply(np.array)\
                        .apply(self.do_t, t = t)
            
            for format_label, format_func in format_conversions:
                edf['%s_pred_%s' % (model, format_label)] = edf.apply(format_func, \
                    field = '%s_pred' % model, pad = False, result_shape=200, axis=1)
                
                for metric_label, metric in metrics:
                    rdf.at[model, (format_label, metric_label)] = edf.apply(lambda row : \
                        metric(row['%s_pred_%s' % (model, format_label)], row[format_label]), axis = 1).mean()
            rdf.at[model, 'support'] = edf[format_label].apply(len).sum()

        self.rdf = rdf
        self.edf = edf

    def do_roc(self):     

        pass 


        # self.rocdf = pd.DataFrame(rdf.T.unstack().T.swaplevel(0,1).groupby(level=0)\
        #                 .apply(lambda x : roc_curve(x.true, x.pred)).to_list(), \
        #                     columns= ['fpr', 'tpr', 'threshold'])

        # self.roc_dict = self.rocdf.apply(lambda row : row.threshold[np.argmax(row.tpr) - \
        #                                             np.argmax(row.fpr)], axis=1 ).to_dict()

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