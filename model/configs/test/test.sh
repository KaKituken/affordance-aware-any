CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node 1 main.py --base configs/test/test_branch.yaml -t --gpus 1 --name test_branch --logdir /n/holyscratch01/pfister_lab/jixuan/logs/test
CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node 1 main.py --base configs/test/test_branch_norm.yaml -t --gpus 1 --name test_branch_normal --logdir /n/holyscratch01/pfister_lab/jixuan/logs/test
CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node 1 main.py --base configs/test/test_pre_fusion.yaml -t --gpus 1 --name test_pre_fusion --logdir /n/holyscratch01/pfister_lab/jixuan/logs/test
CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node 1 main.py --base configs/test/test_mix.yaml -t --gpus 1 --name test_mix --logdir /n/holyscratch01/pfister_lab/jixuan/logs/test

CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node 1 main.py --base configs/dual/dual_branch_rand_conv.yaml -t --gpus 1 --name test_batch --logdir /n/holyscratch01/pfister_lab/jixuan/logs/test
CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node 1 main.py --base configs/test/test_input_branch.yaml -t --gpus 1 --name test_input_branch --logdir /n/holyscratch01/pfister_lab/jixuan/logs/test