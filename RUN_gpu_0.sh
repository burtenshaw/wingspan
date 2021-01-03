# CUDA_VISIBLE_DEVICES=0 python train_bert_ngram.py --method_name bert_ngram --hparams --runs 10
# CUDA_VISIBLE_DEVICES=0 python train_bert_span.py --method_name bert_span
# # CUDA_VISIBLE_DEVICES=0 python train_lstm_ngram.py --method_name lstm_ngram
CUDA_VISIBLE_DEVICES=0 python train_categorical.py --method_name end --hparams --runs 10
CUDA_VISIBLE_DEVICES=0 python train_categorical.py --method_name start --hparams --runs 10
CUDA_VISIBLE_DEVICES=0 python train_categorical.py --method_name spans --hparams --runs 10
CUDA_VISIBLE_DEVICES=0 python train_categorical.py --method_name len --hparams --runs 10