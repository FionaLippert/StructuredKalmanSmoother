#!/bin/bash

cd $HOME/StructuredKalmanSmoother/scripts

which python

export WANDB_MODE="offline"

OUTDIR="/home/flipper/StructuredKalmanSmoother/test_output"
WANDB_RUNS="/home/flipper/StructuredKalmanSmoother/wandb_runs"

mkdir -p $WANDB_RUNS

export WANDB_DIR=$WANDB_RUNS

python test_slurm.py output_dir=$OUTDIR > out.txt 2>&1

#WANDB_RUN=$(awk -F'wandb: wandb sync ' '/wandb: wandb sync/{print $2}' out.txt)

echo $WANDB_DIR

#wandb sync $WANDB_RUN
#wandb sync --sync-all
wandb sync --include-offline $WANDB_DIR/wandb/offline-*