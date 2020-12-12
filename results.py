import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_curve

from utils import *

class EvalResults:

    def __init__(self, results, y_test, roc=False, bin_t=0.5, do_labels=False, token_type='char'):

        rdf = pd.concat([pd.DataFrame(results),pd.DataFrame(y_test)], \
                    keys=['pred', 'true'], axis=1)
        
        self.probsdf = rdf.copy(deep=True)

        if roc:
            self.rocdf = pd.DataFrame(rdf.T.unstack().T.swaplevel(0,1).groupby(level=0)\
                            .apply(lambda x : roc_curve(x.true, x.pred)).to_list(), \
                                columns= ['fpr', 'tpr', 'threshold'])

            roc_dict = self.rocdf.apply(lambda row : row.threshold[np.argmax(row.tpr) - \
                                                        np.argmax(row.fpr)], axis=1 ).to_dict()
            rdf.pred = rdf.pred.apply(lambda col : np.where(col > roc_dict[col.name], 1, 0 ))
        else:
            rdf.pred = rdf.pred.apply(lambda col : np.where(col > bin_t, 1, 0 ))
        
        if token_type == 'char':
            rdf['pred'] = rdf.pred.apply(character_mask_to_entities, axis=1, result_type='expand')
            rdf['true'] = rdf.true.apply(character_mask_to_entities, axis=1, result_type='expand')
        elif token_type == 'word':
            rdf['pred'] = rdf.pred.apply(word_mask_to_character_entity, result_shape=results.shape[1]\
                , axis=1, result_type='expand')
            rdf['true'] = rdf.true.apply(word_mask_to_character_entity, result_shape=results.shape[1]\
                , axis=1, result_type='expand')
        
        self.rdf = rdf
        self.evdf = pd.DataFrame()

        self.evdf['f1'] = rdf.apply(lambda row : \
            f1(row.pred[row.pred != 0 ], row.true[row.true != 0]), axis=1)
        self.evdf['precision'] = rdf.apply(lambda row : \
            precision(row.pred[row.pred != 0 ], row.true[row.true != 0]), axis=1)
        self.evdf['recall'] = rdf.apply(lambda row : \
            recall(row.pred[row.pred != 0 ], row.true[row.true != 0]), axis=1)
        self.mean = self.evdf.mean()

        if do_labels:
            self.label_scores()
        
    
    def label_scores(self):
        
        metrics = ['precision','recall','F1','support']
        prdf = pd.DataFrame(self.rdf.apply(lambda x : precision_recall_fscore_support(x.true, x.pred), axis=1)\
                .apply(lambda x : {m : {n:v for n, v in enumerate(s)} for m,s in zip(metrics, x)})\
                .to_list()).stack().apply(pd.Series)

        prdf['metric'] = prdf.index.get_level_values(1)
        prdf = prdf.groupby('metric').agg(np.mean).T

        avdf = pd.concat([self.rdf.apply(lambda x : \
                    precision_recall_fscore_support(x.true, x.pred, average=av),\
                    axis=1, result_type='expand').mean().to_frame().T.rename(index={0:av}) \
                    for av in ['macro', 'micro', 'weighted']])
        
        avdf.columns= metrics
        
        self.label_scores = pd.concat([prdf,avdf], names=[0,1]).drop(columns='support')
        self.avdf = avdf
        self.prdf = prdf