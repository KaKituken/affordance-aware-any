CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/final/dino.yaml -t --gpus 2 --name dino --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full
CUDA_VISIBLE_DEVICES=2,3, torchrun --nproc_per_node 2 --master_port 25641 main.py --base configs/final/dino_l2.yaml -t --gpus 2 --name dino_l2 --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/final/dino_residual.yaml -t --gpus 2 --name dino_residual --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full
CUDA_VISIBLE_DEVICES=2,3, torchrun --nproc_per_node 2 --master_port 25641 main.py --base configs/final/dino_gated.yaml -t --gpus 2 --name dino_gated --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/final/dino_hiera.yaml -t --gpus 2 --name dino_hiera --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full
CUDA_VISIBLE_DEVICES=2,3, torchrun --nproc_per_node 2 --master_port 25641 main.py --base configs/final/dino_hiera_no_bg.yaml -t --gpus 2 --name dino_hiera_no_bg --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full
CUDA_VISIBLE_DEVICES=2,3, torchrun --nproc_per_node 2 --master_port 25641 main.py --base configs/final/dino_multiscale_3.yaml -t --gpus 2 --name dino_multi_3 --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node 2 main.py --base configs/final/multi_dino.yaml -t --gpus 2 --name multi_dino --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full
CUDA_VISIBLE_DEVICES=0,1, torchrun --master_port 25641 --nproc_per_node 2 main.py --base configs/final/full_baseline.yaml -t --gpus 2 --name full_baseline --logdir /n/holyscratch01/pfister_lab/jixuan/logs/full