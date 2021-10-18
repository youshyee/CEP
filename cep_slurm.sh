#!/bin/bash -login
#SBATCH --job-name=cep

#SBATCH --ntasks=8
#SBATCH --time=28-00:00:00

#SBATCH --mem=120G
#SBATCH --mail-type=ALL
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8

PARTITION=gpu
JOB_NAME=cep_r18

TRAINFILE=./train.py
CONFIG=./configs/kinetics_r18_pretrain.py
GPUS=2

GPUS_PER_NODE=2
CPUS_PER_TASK=8
SRUN_ARGS=${SRUN_ARGS:-""}

while true;
do
srun -p ${PARTITION}\
	--job-name=${JOB_NAME}\
	--gres=gpu:${GPUS_PER_NODE}\
	--ntasks=${GPUS}\
	--ntasks-per-node=${GPUS_PER_NODE}\
	--cpus-per-task=${CPUS_PER_TASK}\
	--kill-on-bad-exit=1\
	${SRUN_ARGS}\
	python -u $TRAINFILE ${CONFIG}  --launcher="slurm"
done

