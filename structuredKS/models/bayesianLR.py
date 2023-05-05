from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

config = {"seed": 0,
          "dataset": "spatiotemporal_20x20_obs=0.7_T=5_diff=0.1_adv=constant_ntrans=1_0",
          # "dataset": "pems_start=0_end=215",
          "noise_std": 0.001,
          "use_bias": False,
          "n_training_samples": 10,
          "n_post_samples": 100,
          "lr": 0.01,
          "val_interval": 100,
          "n_iterations": 1000,
          "tol": 1e-7}

seed_all(config['seed'])