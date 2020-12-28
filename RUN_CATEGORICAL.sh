
tensorboard --logdir logs/categorical --port 6009 & \
CUDA_VISIBLE_DEVICES=0 python train_categorical.py start & \
CUDA_VISIBLE_DEVICES=1 python train_categorical.py end 

CUDA_VISIBLE_DEVICES=0 python train_categorical.py spans & \
CUDA_VISIBLE_DEVICES=1 python train_categorical.py len

