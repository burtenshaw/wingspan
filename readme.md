# TODO

## preprocessing
- [x] word mask back to contiguous ranges

## models
- [x] use all task specific evaluations
- [x] try more models
- [x] try ensemble
- [x] add lexical
- [x] add word chunk prediction; predict if word is toxic based on last 4 word_mask
- [x] switch muse embeddings to use
- [ ] try electra on mask prediction

## categorical sub-tasks
- [x] add multilabel as embedding
- [x] predicted n_spans, 
- [x] predicted multilabel toxicity, 
- [x] predict span start,
- [x] predicted span end,
- [x] predicted span length
- [x] spacy vectors 
- [x] spacy word sentiment 
- [x] glove word embeddings
- [x] sentence bert
- [ ] train an ensemble model using numerical data

## Postprocessing

- [x] improve accuracy from mask to contiguous ranges is f1 0.92 

## Cross validation_data

- [x] 5 fold cross validation

# Current Methods :

## lexical 

- predict word level toxicity based on a list of toxic words
- 0.4 on real task

## ngram_glove_lstm

- take ngrams of before and after word n and predict if n = toxic
- performing well for simplicity
- added separate word embedding
- trying on longer ngrams
- TODO : add word embedding to hp optimisation loop    
- 0.8 on word toxic prediction

## ngram_bert

- as above using bert
- under performing on shorter ngrams
- trying on longer ngrams
  overfitting - redone hparams optimisation 

## mask_bert

- predict the word toxicity given a sentence of a sentence given, use token type_ids to mark the target word
- flawed - redo with ne mask for each word to predict

## span_bert

- predict the binary word mask of a sentence
- applied up sampling
- TODO : find a way of weighting word labels over binary word mask target
- TODO : add up sampling into hp optimisation loop

# Categorical Sub methods

## predicted mutilabel toxicity

## predicted n_spans

`categorical_spans`

- predict number of spans [ to be ensembled with other numerical tasks]
- ~ 0.97 on nspan prediction
- overfitting on 1 - add class weights

## predict span start

## predicted span end

## predicted span length

## train a model using numerical data

## ensemble
