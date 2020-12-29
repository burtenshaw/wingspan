
tensorboard --logdir logs/categorical --port 6009 & \
CUDA_VISIBLE_DEVICES=0 python train_categorical.py start & \
CUDA_VISIBLE_DEVICES=1 python train_categorical.py end 

CUDA_VISIBLE_DEVICES=0 python train_categorical.py spans & \
CUDA_VISIBLE_DEVICES=1 python train_categorical.py len

tensorboard --logdir logs/span --port 6010 & \
tensorboard --logdir logs/ngram --port 6011 & \
CUDA_VISIBLE_DEVICES=1 python train_bert_ngram.py bert_ngram & \
CUDA_VISIBLE_DEVICES=0 python train_bert_span.py bert_span

# CUDA_VISIBLE_DEVICES=0 python train_lstm_ngram.py lstm_ngram


