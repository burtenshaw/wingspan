
fold () {

    local run=$1
    tensorboard --logdir logs/categorical --port 6009 & \
    tensorboard --logdir logs/span --port 6010 & \
    tensorboard --logdir logs/ngram --port 6011 & \
    CUDA_VISIBLE_DEVICES=0 python train_categorical.py --method_name start --fold run & \
    CUDA_VISIBLE_DEVICES=1 python train_categorical.py --method_name end --fold run 
    CUDA_VISIBLE_DEVICES=0 python train_categorical.py --method_name spans --fold run & \
    CUDA_VISIBLE_DEVICES=1 python train_categorical.py --method_name len --fold run
    CUDA_VISIBLE_DEVICES=0 python train_bert_ngram.py --method_name bert_ngram --fold run & \
    CUDA_VISIBLE_DEVICES=1 python train_bert_span.py --method_name bert_span --fold run
    CUDA_VISIBLE_DEVICES=0 python train_lstm_ngram.py --method_name lstm_ngram --fold run
}

local runlist = '0 1 2 3 4'

for run in runlist; do fold "$run" & done



