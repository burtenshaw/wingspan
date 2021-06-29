# Introduction

#### SemEval 2021 Task 5:

Toxic Spans Detection was organised by John Pavlopoulos and colleagues, and described in detail in their task description paper. Competing teams were asked to develop systems capable of detecting spans of toxic text. Predictions were evaluated using a pairwise F1-score of toxic character offset predictions.

#### Initial analysis

Initial analysis of the development data revealed that toxic spans were varied in content and not limited to single words. Though most examples contained single toxic words or phrases, others contained longer spans and complete sentences. With this in mind, we sort a strategy that combined longer span based detection with binary word classification.

#### Strategy

We combined models that used antithetical contexts, i.e. full sequences, and shorter ngram sequences before and after a given word. This approach is based on the hypothesis that their predictions would have a low correlation, and in turn, they would create ideal ensemble components.

#### Results

The system described in this paper scored **0.6755** and ranked **26^th^**. We discovered that model correlation did play a factor in the accuracy of an ensemble approach; however, much of this performance increase was lost in transition to test data, where correlation increased on the most frequent type of examples. We analyse model performance and correlation in relation to textual features.

#### Features

Teams were supplied with development data consisting of 7939 text samples in varying lengths up to 1000 characters, and tested on 2000 text samples.

# System overview

![Model Diagram including all component models. Colours represent layer types and arrows represent training pipeline. ](figures/ensemble_plot.png)

#### Task Interpretations

We used two types of component models; binary word level models and categorical span based models, and combined those in an LSTM network [@hochreiter1997long]. We used two word based models \[GLOV, BERT\] and three span based models \[ALBE, ROBE, ELEC\], the softmax output of all models were concatenated and supplied to an LSTM model \[ENSE\].

#### Motivation

We intended for the word based models to learn local features in the tokens nearest the target word, and for the span based to learn the overall features that affected sub and multi word toxicity.

## Baselines {#Baseline-Models}

To interpret the task we relied on the Spacy implemented baseline shared by the organizers and described in the task description paper [@johnpavlopoulosSemEval2021Task2021; @spacy]. The approach retrained the RoBERTa based `en_core_web_trf` model's `ner`, `trf_wordpiecer`, and `trf_tok2vec` components, producing f1-scores of 0.5630 on the development data and 0.6305 on test data. To Interpret the problem further, we implemented two simple baselines.

#### Lexical Lookup {#Lexical}

Using a subset of samples from the development data, we created a toxic words list from all words within toxic spans, except for stop words [^1]. On the test data, we then classified words as toxic if they appeared within the aforementioned toxic words list. We then converted word offsets into character offsets. This approach achieved an F1-score of 0.4161 on the test data.

#### SVM

Using Term Frequency to Inverse Document Frequency we created two document vector representations of toxic and non-toxic spans. Using a Support Vector Machine, we predicted the probability that a word vector appeared within a toxic or non-toxic document [@salton1986introduction; @wuProbabilityEstimatesMulticlass]. We then used a binary threshold of 0.5 and class weights based on relative label frequency to predict whether a word was toxic. This approach achieved an F1-score of 0.5489 on the test data.

## Component Models {#Component-Models}

### Span Prediction {#Span-Prediction}

![Illustration of toxic span prediction based on complete sequence.](figures/span_example.png){#fig:ngram-example width="\\columnwidth"}

Span prediction models used the complete sequence of words, up to a maximum length, to predict toxic character offsets. Sequences were represented as token reference indexes, described in section [4.1.0.1](#trans-process){reference-type="ref" reference="trans-process"}. The target sequence was processed from character offsets into categorical arrays for toxic, non-toxic, and padding tokens. [4.1.0.2](#target-labels){reference-type="ref" reference="target-labels"}.

#### Transformer Models {#albert}

[\[Electra\]]{#Electra label="Electra"}[\[RoBERTa\]]{#RoBERTa label="RoBERTa"}

We selected three pretrained transformer models (ALBERT, RoBERTa, ELECTRA) and fine-tuned them for this task with extra linear layers. We performed separate hyper-parameter optimisation for each model, detailed in section [4.2.0.2](#hp-opt){reference-type="ref" reference="hp-opt"}. ALBERT is a lightweight implementation of a BERT model [@lanALBERTLiteBERT2020; @devlin-etal-2019-bert] that uses feature reduction to reduce training time. ELECTRA is a further development of the BERT model that pre-trains as a discriminator rather than a generator [@clarkELECTRAPRETRAININGTEXT2020]. RoBERTa develops the BERT model approach for robustness, [@liuRoBERTaRobustlyOptimized2019]. During development we found that these three transformer models achieved the highest f1-scores in relation model correlation compared to alternatives. All models used the Adam optimizer [@kingmaAdamMethodStochastic2017].

### Binary Word Prediction {#Binary-Prediction}

![Illustration of toxic word prediction based on sequence before and after target word.](figures/ngram_example.png){#fig:ngram-example width="\\columnwidth"}

The binary word level models treated the task as word toxicity prediction based on a sequences of words before and after the target word. Figure [3](#fig:ngram-example){reference-type="ref" reference="fig:ngram-example"} illustrates this approach. The target word toxicity was represented as a binary value. The sequence length before and after the target word was optimised for each model, and described in section [4.2.0.2](#hp-opt){reference-type="ref" reference="hp-opt"}.

#### Siamese-LSTM with Glove Word Embeddings {#LSTM-Glove}

A Siamese LSTM model used two networks based on separate glove embeddings of the sequence of words before and after the target words [@baoAttentiveSiameseLSTM2018; @baziotisDataStoriesSemEval2017Task2017].

#### LSTM Finetuning BERT-base {#Bert-base}

An LSTM model was trained based on the output of a BERT-base model. The words before and after the target word were used as model features, and the target word toxicity was represented as a binary value [@devlin-etal-2019-bert].

## Ensemble Model {#Ensemble-Model}

A Bidirectional LSTM model was used to predict token toxicity based on tokenised word features and component model predictions. The model used transformer style feature representations to predict a sequence of categorical representations for token toxicity, as described in section [4.1.0.2](#target-labels){reference-type="ref" reference="target-labels"}. The ensemble model relied on five fold cross validation, as described in section [4.2.0.1](#cross-val){reference-type="ref" reference="cross-val"}.

### Component model Predictions

Component model predictions were concatenated together as categorical representations of labels (not toxic, toxic, padding : 0,1,2). Each model's 3 dimensional output (number of samples, sequence length, number of labels) was permuted into a 4 dimensional matrix (number of samples, sequence length, number of labels, number of models).

# Experimental setup

## Pre-Processing

#### Tokenisation {#trans-process}

Text sequences were tokenised into character sequences using a BERT tokenizer and excess characters were replaced with a `#` character, as shown in Figure [\[fig:bert-tokens\]](#fig:bert-tokens){reference-type="ref" reference="fig:bert-tokens"} [@devlin-etal-2019-bert]. Sequences were padded and truncated for uniformity to a length of 200 tokens. Longer sequences were handled separately, and predictions were combined in post-processing, described in section [4.4](#post){reference-type="ref" reference="post"}.

------------------------- -------------------------------------------------------------- ----- ------- --------------------------------------- -------------------------------------------

#### Target Label Representation {#target-labels}

To best suit the component models, we used a target representation based on the character sequences from the BERT tokenizer. Each word-like sequence was given a label based on its `word-id`, and converted into categorical binary arrays, or one-hot vectors.

## Training and Optimisation

#### Cross Validation {#cross-val}

We used stratified $k$ fold validation of the development data to train all models. After optimisation, each component model's predictions on the *test* portion of fold $k$ were added to the *train* portion of the other folds. Producing unseen training features for the ensemble model. This process avoids overfitting in component models, and facilitates training an ensemble model on the complete development data [@fushiki2011estimation; @pedregosa2011scikit].

#### Hyper-Parameter Optimisation {#hp-opt}

Model parameters were optimised for each fold of the development data and the best models were used by the ensemble model. Table [2](#tab:best-params){reference-type="ref" reference="tab:best-params"} shows the optimum parameters for each model used on the test data. We used Bayesian optimization for each fold of the development data to find optimum parameters [@snoekPracticalBayesianOptimization]. Component models were selected based on their f1-score and prediction correlation to other models.

## Prediction

To predict spans for submission, a version of each component model optimised for each fold of the development data was supplied the test data and their outputs were averaged. The ensemble model was then supplied component model predictions and tokenised text sequences.

## Post-processing {#post}

Model output was converted from 2 dimensional token-level categorical arrays ($n$ tokens, $n$ labels) into character offsets. The character offsets of each positively labeled token was then added to a list. The predictions of sequences that had been truncated during pre-processing, were combined and duplicates were removed.

# Results

![results](figures/results_table.png)

The table reveals that the ensemble model achieved a similar score on both development and test data, while the ALBERT, ELECTRA, and baseline models improved in testing. Crucially, the $\widetilde5\%$ increase in f1-score from component models to ensemble, that we see on the development data, was not transferred to the test data.

## Model Correlation {#model-corr}

Figure [4](#fig:corr){reference-type="ref" reference="fig:corr"} reveals that the ensemble and ALBERT models have a high correlation, a logical outcome of their shared base layers; whilst word based models \[BERT, GLOV\] have a low correlation, reflecting their diverse interpretations.

![Model Correlation calculated using a macro average f1-score](figures/corr.png){#fig:corr width="\\columnwidth"}

## Error Analysis

We performed error analysis to interpret the hypothesis that there are multiple annotation rationales; single toxic words, and longer offensive sentences, illustrated in Figure [\[fig:different-spans\]](#fig:different-spans){reference-type="ref" reference="fig:different-spans"}.

#### Toxic Span Length

Figure [5](#fig:len){reference-type="ref" reference="fig:len"} reveals that the length of toxic spans had an impact on model performance. Models were less accurate at detecting longer spans on both development and test data. Furthermore, the impact of this effect on test data was decreased as there were fewer longer toxic spans.

![Model F1 score at $n$ tokens per toxic span. Bars show the frequency of $n$ tokens in development and test data. Shaded areas shows standard deviation of the f1-score for the ensemble model.](figures/performance_span_len.png){#fig:len width="\\columnwidth"}

#### Stop Words in Toxic Spans

The frequency of stop words in toxic spans also affected model performance. Figure [6](#fig:stop){reference-type="ref" reference="fig:stop"} reveals that, where present, spans with more stop words caused lower model accuracy.

![Model F1 score at $n$ stop words per toxic span, and $n$ stop word frequency.](figures/performance_stop_words.png){#fig:stop width="\\columnwidth"}

#### Binary Token Level Evaluation {#word-based}

By using token level scoring we are able to reveal how the models perform on both positive and negative tokens. Here, the target labels are represented as binary arrays; 1 for toxic tokens and 0 for non-toxic. We can not expect these calculations to align with character offsets, due to variance in tokenisation and parsing.

![Binary token level scores for precision, recall, and f1-score.](figures/word_f1_matrix.png)

# Conclusion

Our initial hypothesis, that combining word based and span based approaches would yield a significant performance boost, did not stand up. We measured a $\widetilde5\%$ increase in f1-score on development data, but this was not transferred to test data. In future work, we would look to a strategy that incorporated model transferability in component model selection, with the intention of better handling fluctuations in annotation rationale. Drawing on recent work [@fortunaHowWellHate2021].

[^1]: The toxic words list was created from the first 5800 samples of the development data. We used Spacy tokenisation and English stop words list, and we removed space and character offsets from predictions.
