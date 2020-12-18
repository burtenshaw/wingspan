# to do

- [x] improve accuracy of word mask back to ranges
- [x] use all task specific evaluations
- [x] try more models
- [x] try ensemble
- [x] add lexical

- [x] add word chunk prediction; predict if word is toxic based on last 4 word_mask
- [x] switch muse embeddings to use
- [ ] add multilabel as embedding
- [ ] ensemble numerical values
        create a df of  : 
            - [x] predicted n_spans, 
            - [ ] predicted toxicity, 
            - [ ] predict span start,
            - [ ] predicted span end,
            - [ ] predicted span length
        - [ ] train a model using numerical data


- [ ] try electra on mask prediction


# current methods :

lexical : 

    - predict word level toxicity based on a list of toxic words
    - 0.4 on real task

ngram_glove_lstm

    - take ngrams of before and after word n and predict if n = toxic
    - performing well for simplicity
    - added seperate word embedding
    - trying on longer ngrams
    - TODO : add word embedding to hp optimisation loop
    
    - 0.8 on word toxic prediction


ngram_bert

    - as above using bert
    - under performing on shorter ngrams
    - trying on longer ngrams

mask_bert

    - predict the word toxicity give a sentence of a sentence given, use token type_ids to mark the target word
    - flawed 

span_bert

    - predict the binary word mask of asentence
    - applied up sampling
    - TODO : find a way of weighting word labels over binary word mask target
    - TODO : add up sampling into hp optimisation loop

categorical_bert

    predict number of spans [ to be ensembled with other numerical tasks]
    ~ 0.97 on nspan prediction

ensemble
