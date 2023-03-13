import torch
import numpy as np
import json
import os
import time
import argparse
import wandb
import copy
from matplotlib import pyplot as plt

# import visualization as vis
from structuredKS.models.dgmrf import DGMRF, SpatialDGMRF, ObservationModel, VariationalDist, vi_loss, posterior_inference
import constants_dgmrf as constants
import utils_dgmrf as utils

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


config = {"seed": 0,
          "n_layers": 1,
          "non_linear": False,
          "fix_gamma": False,
          "log_det_method": "eigvals",
          "use_bias": False}


seed_all(0)

if torch.cuda.is_available():
    # Make all tensors created go to GPU
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # For reproducability on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dataset_dict = utils.load_dataset("spatiotemporal_20x20_T=3_diff=0.0_adv=None_0")
graphs = dataset_dict["graphs"]

M = graphs["data"].num_nodes
N = graphs["latent"].num_nodes

graph_0 = graphs.get_example(0)
graph_t = graphs.get_example(1)

dgmrf = SpatialDGMRF(graph_0["latent", "spatial", "latent"], graph_t["latent", "spatial", "latent"], config)

# test dgmrf
x = graphs["latent"].x
Gx = dgmrf(graphs)
prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) + dgmrf.log_det(len(graphs))) / N

vi_entropy = 0

obs_model = ObservationModel()
y_hat = obs_model(x, graphs["latent", "observation", "data"])
y = graphs["data"].x

noise_std = graphs.get_example(0)['data'].noise_std
log_noise_std = torch.log(noise_std)
data_ll = -0.5 * torch.sum(torch.pow((y - y_hat), 2)) / (M * noise_std**2) - log_noise_std

elbo = prior_ll + vi_entropy + data_ll
print(prior_ll, data_ll)