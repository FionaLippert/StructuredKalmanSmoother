#!/bin/bash

cd $HOME/StructuredKalmanSmoother

which python

export WANDB_MODE="offline"
export WANDB_DIR="wandb_runs"

python scripts/test_slurm.py > out.txt 2>&1

#WANDB_RUN=$(awk -F'wandb: wandb sync ' '/wandb: wandb sync/{print $2}' out.txt)

#wandb sync $WANDB_RUN
wandb sync --sync-all