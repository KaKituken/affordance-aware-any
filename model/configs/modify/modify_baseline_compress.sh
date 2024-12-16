#!/bin/bash
#SBATCH -c 32
#SBATCH --time 3-0          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --mem-per-cpu 7168  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:2       # Number of gpu
#SBATCH -o /n/home11/jxhe/insert-any/insert_anything/sd_style/logs/baseline_compress.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home11/jxhe/insert-any/insert_anything/sd_style/logs/baseline_compress.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=838242595@qq.com

mamba init
mamba activate putany

CUDA_VISIBLE_DEVICES=0,1, \
python /n/home11/jxhe/insert-any/insert_anything/sd_style/main.py \
    --base /n/home11/jxhe/insert-any/insert_anything/sd_style/configs/modify/modify_baseline_compress.yaml \
    -t \
    --gpus 2 \
    --name baseline_compress \
    --logdir /n/holyscratch01/pfister_lab/jixuan/logs/modify

CUDA_VISIBLE_DEVICES=0
python /n/home11/jxhe/insert-any/insert_anything/sd_style/main.py \
    --base /n/home11/jxhe/insert-any/insert_anything/sd_style/configs/modify/modify_baseline_compress.yaml \
    -t \
    --gpus 1 \
    --name baseline_compress \
    --logdir /n/holyscratch01/pfister_lab/jixuan/logs/modify