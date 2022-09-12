#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH--partition=small

# module load python/anaconda3
# conda env create -f environment.yml
# source activate AL

# Load environment
# add module cuda
module load cuda/10.1

# add conda load
module load python/anaconda3
source activate AL

# Run Script
TRANSFORMERS_OFFLINE=1 \
python SL_transformers.py --method ${METHOD} --framework ${FRAMEWORK} \
--datadir ${DATADIR} --dataset ${DATASET} --outdir ${OUTDIR} \
--transformer_model ${TRANSFORMER_MODEL}  \
--n_epochs ${N_EPOCHS} --class_imbalance ${CLASS_IMBALANCE} \
--train_n ${TRAIN_N} --test_n ${TEST_N} --run_n ${RUN_N} \
> ${OUTDIR}/${METHOD}_${FRAMEWORK}_${DATASET}_${CLASS_IMBALANCE}_${TRAIN_N}/slurm_log 2> ${OUTDIR}/${METHOD}_${FRAMEWORK}_${DATASET}_${CLASS_IMBALANCE}_${TRAIN_N}/slurm_err

