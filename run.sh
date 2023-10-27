# for single gpu
<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=1 python3.8 train2.py --cfg ./configs/pa100k.yaml
=======
CUDA_VISIBLE_DEVICES=1 python3.8 train_classifier.py --cfg ./configs/pa100k.yaml

# for multi-gpu
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1233 train.py --cfg ./configs/pa100k.yaml
>>>>>>> ba6ce9aa78f1ce5d1680af52e2bd8d4b1be62ae9
