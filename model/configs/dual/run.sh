CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/dual/dual.yaml -t --gpus 2 --name dual --logdir /n/holyscratch01/pfister_lab/jixuan/logs/dual
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/dual/dual_5e-4.yaml -t --gpus 2 --name dual_5e-4 --logdir /n/holyscratch01/pfister_lab/jixuan/logs/dual
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/dual/dual_2.5e-4.yaml -t --gpus 2 --name dual_2.5e-4 --logdir /n/holyscratch01/pfister_lab/jixuan/logs/dual
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/dual/dual_10k.yaml -t --gpus 2 --name dual_10k --logdir /n/holyscratch01/pfister_lab/jixuan/logs/dual
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/dual/dual_7.5e-4.yaml -t --gpus 2 --name dual_7.5e-4 --logdir /n/holyscratch01/pfister_lab/jixuan/logs/dual