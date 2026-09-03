#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time=0-00:15:00

#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# #SBATCH --gpus-per-task=h100:4
#SBATCH --gres=gpu:h100:4 # per node

#SBATCH --cpus-per-task=8 # please count CPUs/GPUs number when used with torchrun (process-per-GPU) 
#SBATCH --mem=32G # per node, please count mem/GPUs when used with torchrun

#SBATCH --output=logs/stdout%j.out
#SBATCH --error=logs/stderr%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alar.akilbekov@nu.edu.kz

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.9.1

mkdir -p logs

# srun bash -c 'nvidia-smi'

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29551

# export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK # Slurm 22
# srun bash -c 'echo "$(hostname): nproc=$(nproc) $(taskset -cp $$)"'

export OMP_NUM_THREADS=1 
# Against oversubscription:
# srun -> node=0 -> task with --cpus-per-task=8 -> torchrun (process) => 
#   (processes) 
#       main.py LOCAL_RANK=0 (process) => num_workers=4 (processes)
#       main.py LOCAL_RANK=1 (process) => num_workers=4 (processes)
#   Each worker with num_threads=~8 (32 threads) or OMP_NUM_THREADS=1 (8 threads)

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=ens9f0 # rendezvous uses TCP/IP: `ip -br addr`
export NCCL_IB_HCA=mlx5_0,mlx5_1 # allreduce uses IB: `ibv_devinfo`
# irdma0 is 25GB/s, IB is 400GB/s

source .venv/bin/activate

# srun -> ~~bash~~ -> torchrun
srun --wait=120  bash -c '
    exec torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$SLURM_GPUS_ON_NODE \
        --node-rank=$SLURM_NODEID \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        main.py
    '
