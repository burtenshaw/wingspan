CUDA_VISIBLE_DEVICES=1 python train_categorical.py --method_name end --fold $run
CUDA_VISIBLE_DEVICES=1 python train_categorical.py --method_name start --fold $run 
CUDA_VISIBLE_DEVICES=1 python train_categorical.py --method_name spans --fold $run
CUDA_VISIBLE_DEVICES=1 python train_categorical.py --method_name len --fold $run
    

