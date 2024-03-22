# Spatiotemporal Deep Gaussian Markov Random Fields (ST-DGMRF)
A computationally efficient approach to state estimation and learning in graph-structured state-space models with 
(partially) unknown dynamics and limited historical data.

For more information, see our paper [Deep Gaussian Markov Random Fields for Graph-Structured Dynamical Systems](
https://openreview.net/pdf?id=dcw7qRUuD8
)

## Installation
To install the `stdgmrf` python package, run
```
python -m pip install --upgrade build
python -m build
pip install .
```

## Training and inference
To train an ST-DGMRF and perform inference, run
```
python scripts/run_stdgmrf.py dataset=<dataset-name>
```
This will use the default settings defined in `scripts/conf/config.yaml`. To change these settings, you can either adjust this file, or change them via the command line (will be parsed with [hydra](https://hydra.cc/docs/intro/)).

## Experiments
We use [Weights & Biases](https://wandb.ai/site) sweeps to perform our experiments.
All relevant config files defining these sweeps can be found in [scripts/experiments](https://github.com/FionaLippert/StructuredKalmanSmoother/tree/main/scripts/experiments). To run an experiment, first initialize the sweep with
```
wandb sweep --project <propject-name> <path-to-config file>
```
and then, using the obtained `agent-ID`, run
```
wandb agent <agent-ID>
```
 
