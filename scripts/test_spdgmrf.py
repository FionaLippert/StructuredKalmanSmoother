import torch
import numpy as np
import json
import os
import time
import argparse
import wandb
import copy
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

print('import KS package stuff')

# import visualization as vis
from structuredKS.models.dgmrf import *
import constants_dgmrf as constants
import utils_dgmrf as utils
from structuredKS.datasets.dummy_dataset import DummyDataset

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

print('define config')

config = {"seed": 0,
          "dataset": "spatiotemporal_20x20_obs=0.7_T=6_diff=0.01_adv=periodic_ntrans=4_0",
          "noise_std": 0.01,
          "n_layers": 1,
          "n_transitions": 1,
          "non_linear": False,
          "fix_gamma": False,
          "log_det_method": "eigvals",
          "use_bias": False,
          "n_training_samples": 10,
          "n_post_samples": 100,
          "vi_layers": 1,
          "features": False,
          "optimizer": "adam",
          "lr": 0.01,
          "data_split": [0.8, 0.2, 0.0],
          "val_interval": 100,
          "n_iterations": 100,
          "use_dynamics": True,
          "independent_time": False,
          "use_hierarchy": False,
          "transition_type": "advection+diffusion",
          "inference_rtol": 1e-7}

class LogPredictionsCallback(Callback):

    def __init__(self, logger):
        self.logger = logger

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        for key, val in outputs.items():
            self.logger.log_image(key=key, images=[img for img in val])


class LatticeInferenceCallback(Callback):

    def __init__(self, logger):
        self.logger = logger

    def masked_image(self, img, mask):
        return wandb.Image(img.cpu().detach().numpy(),
                           masks={'observations': {'mask_data': mask.cpu().detach().numpy(),
                                                   'class_labels': {0: "unobserved", 1: "observed"}}})

    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, 'gt'):
            img_shape = (pl_module.T, pl_module.grid_size, pl_module.grid_size)
            gt = pl_module.gt.reshape(img_shape)
            masks = pl_module.mask.reshape(img_shape)
            # self.logger.log_image(key='gt', images=[self.masked_image(gt[t], masks[t]) for t in range(pl_module.T)])
            self.logger.log_image(key='gt', images=[img for img in gt])
            self.logger.log_image(key='mask', images=[img.float() for img in masks])

    def on_train_epoch_start(self, trainer, pl_module):
        # plot current vi mean and std
        img_shape = (pl_module.T, pl_module.grid_size, pl_module.grid_size)
        mean, std = pl_module.vi_dist.posterior_estimate()
        mean = mean.reshape(img_shape)
        std = std.reshape(img_shape)

        self.logger.log_image(key='vi_mean', images=[img for img in mean])
        self.logger.log_image(key='vi_std', images=[img for img in std])

        if hasattr(pl_module, 'vi_dist_vx'):
            mean_vx = pl_module.vi_dist_vx.posterior_estimate()[0].reshape(pl_module.grid_size, pl_module.grid_size)
            mean_vy = pl_module.vi_dist_vy.posterior_estimate()[0].reshape(pl_module.grid_size, pl_module.grid_size)

            self.logger.log_image(key='vi_v_mean', images=[mean_vx, mean_vy])

    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, 'gt'):
            img_shape = (pl_module.T, pl_module.grid_size, pl_module.grid_size)
            gt = pl_module.gt.reshape(img_shape).cpu().detach()

            # VI inference
            mean, std = pl_module.vi_dist.posterior_estimate()
            mean = mean.reshape(img_shape).cpu().detach()
            std = std.reshape(img_shape).cpu().detach()

            # CG inference
            if config['use_hierarchy']:
                v_mean = torch.stack([pl_module.vi_dist_vx.mean_param, pl_module.vi_dist_vy.mean_param], dim=0).unsqueeze(1)
                print(v_mean.size())
            else:
                v_mean = None
            data = torch.zeros(pl_module.mask.size())
            data[pl_module.mask] = pl_module.y
            true_mean = posterior_mean(pl_module.dgmrf, data.reshape(1, *pl_module.input_shape),
                                       pl_module.mask, config, v=v_mean).reshape(img_shape).cpu().detach()

            data = torch.ones_like(pl_module.mask) * np.nan
            data[pl_module.mask] = pl_module.y
            data = data.reshape(img_shape).cpu().detach()

            residuals = gt - mean

            vmin = gt.min()
            vmax = gt.max()

            fig, ax = plt.subplots(6, pl_module.T, figsize=(pl_module.T * 8, 6 * 8))
            for t in range(pl_module.T):
                # data
                ax[0, t].imshow(data[t], vmin=vmin, vmax=vmax)
                ax[0, t].set_title(f't = {t}', fontsize=30)
                ax[0, 0].set_ylabel('data', fontsize=30)

                # ground truth
                data_img = ax[1, t].imshow(gt[t], vmin=vmin, vmax=vmax)
                ax[1, 0].set_ylabel('ground truth', fontsize=30)

                # true posterior mean
                ax[2, t].imshow(true_mean[t], vmin=vmin, vmax=vmax)
                ax[2, 0].set_ylabel('true posterior mean', fontsize=30)

                # VI posterior mean
                ax[3, t].imshow(mean[t], vmin=vmin, vmax=vmax)
                ax[3, 0].set_ylabel('VI mean', fontsize=30)

                # VI posterior std
                std_img = ax[4, t].imshow(std[t], vmin=std.min(), vmax=std.max(), cmap='Reds')
                ax[4, 0].set_ylabel('VI std', fontsize=30)

                # residuals
                res_img = ax[5, t].imshow(residuals[t], vmin=-residuals.abs().max(), vmax=residuals.abs().max(),
                                          cmap='coolwarm')
                ax[5, 0].set_ylabel('residuals', fontsize=30)

            cbar1 = fig.colorbar(data_img, ax=ax[:4, :], shrink=0.6, aspect=10)
            cbar2 = fig.colorbar(std_img, ax=ax[4, :], shrink=0.8, aspect=4)
            cbar3 = fig.colorbar(res_img, ax=ax[5, :], shrink=0.8, aspect=4)

            cbar1.ax.tick_params(labelsize=20)
            cbar2.ax.tick_params(labelsize=20)
            cbar3.ax.tick_params(labelsize=20)

            wandb.log({'overview': wandb.Image(plt)})
            plt.close(fig)

print('set seed')

seed_all(0)

print('setup cuda')

if torch.cuda.is_available():
    # Make all tensors created go to GPU
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # For reproducability on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print('setup wandb')

# Init wandb
wandb_name = f"{config['dataset']}-{config['transition_type']}-{time.strftime('%H-%M')}"
wandb.init(project=constants.WANDB_PROJECT, config=config, name=wandb_name)

print('load data')

dataset_dict = utils.load_dataset(config["dataset"])
if config["use_dynamics"]:
    graphs = dataset_dict["spatiotemporal_graphs"]
    print(graphs)
else:
    graphs = dataset_dict["spatial_graphs"]

M = graphs["data"].num_nodes
N = graphs["latent"].num_nodes
T = len(graphs)

print(graphs["latent"])
print(graphs["data"])
print(N, M)

print('define models')

# initial_guess = (graphs["latent"].y * graphs["latent"].mask).reshape(T, -1)
print(f'initial guess = {graphs["data"].x.mean()}')
initial_guess = torch.ones(N).reshape(T, -1) * graphs["data"].x.mean()
model = SpatiotemporalInference(graphs, initial_guess, config)

# dataloaders contain data masks defining which observations to use for training, validation, testing
n_train = int(M * config["data_split"][0])
n_val = int(M * config["data_split"][1])
n_test = M - n_train - n_val
random_idx = torch.randperm(M)
ds_train = DummyDataset(random_idx[:n_train], config["val_interval"])
ds_val = DummyDataset(random_idx[n_train:n_train+n_val], 1)
# ds_test = DummyDataset(random_idx[-n_test:], 1)
ds_test = DummyDataset(torch.zeros(1), 1)
dl_train = DataLoader(ds_train, batch_size=1, shuffle=False)
dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
dl_test = DataLoader(ds_val, batch_size=1, shuffle=False)

wandb_logger = WandbLogger(log_model='all')
# log_predictions_callback = LogPredictionsCallback(wandb_logger, t=1)
inference_callback = LatticeInferenceCallback(wandb_logger)

# wandb_logger.log_image(key='initial_guess',
#         images=[img.reshape(model.grid_size, model.grid_size) for img in initial_guess])
# wandb_logger.log_image(key='observation_mask',
#         images=[graphs.get_example(t)["latent"].mask.reshape(model.grid_size, model.grid_size).float() for t in range(T)])

trainer = pl.Trainer(
    max_epochs=int(config["n_iterations"] / config["val_interval"]),
    log_every_n_steps=1,
    logger=wandb_logger,
    deterministic=True,
    accelerator='gpu',
    devices=1,
    callbacks=[inference_callback],
)

trainer.fit(model, dl_train, dl_val)
trainer.test(model, dl_test)


# if hasattr(model.dgmrf, 'dynamics'):
#     if hasattr(model.dgmrf.dynamics.transition_model, 'diff_coeff'):
#         print(f'diff estimate = {model.dgmrf.dynamics.transition_model.diff_coeff}')
#     if hasattr(model.dgmrf.dynamics.transition_model, 'velocity'):
#         print(f'velocity estimate = {model.dgmrf.dynamics.transition_model.velocity}')


# dgmrf = ParallelDGMRF(graphs, config)
# opt_params = tuple(dgmrf.parameters())
#
# # Train using VI
# vi_dist = ParallelVI(graphs, initial_guess, config)
# opt_params += tuple(vi_dist.parameters())
#
# opt = utils.get_optimizer(config["optimizer"])(opt_params, lr=config["lr"])
# total_loss = torch.zeros(1)

# print('start iterations')
#
# for iteration_i in range(config["n_iterations"]):
#     opt.zero_grad()
#
#     print(f'iteration {iteration_i}')
#
#     # sample from q(x)
#     print('sample from q(x)')
#     samples = vi_dist.sample() # shape [T, n_samples, num_nodes]
#
#     print('compute g(x)')
#     # compute log-likelihood of samples given prior p(x)
#     Gx = dgmrf(samples) # shape (T, n_samples, n_nodes)
#     prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) + dgmrf.log_det()) / N
#
#
#     print('compute p(y given x)')
#     # compute data log-likelihood given samples
#     obs_model = ObservationModel()
#     print(f'sample size = {samples.size()}', T, N)
#     samples = samples.transpose(0, 1) # shape [n_samples, T, num_nodes]
#     samples = samples.reshape(config["n_training_samples"], -1) # shape [n_samples, T * num_nodes]
#     y_hat = obs_model(samples, graphs["latent", "observation", "data"])
#     print(f'y_hat size = {y_hat.size()}')
#
#     y = graphs["data"].x
#     noise_std = graphs.get_example(0)['data'].noise_std
#     log_noise_std = torch.log(noise_std)
#     data_ll = -0.5 * torch.sum(torch.pow((y - y_hat), 2)) / (M * noise_std**2) - log_noise_std
#
#     vi_entropy = 0.5 * vi_dist.log_det() / N
#
#     print('compute loss')
#     loss = - (prior_ll + vi_entropy + data_ll)
#
#
#     # Train
#     loss.backward()
#     opt.step()
#
#     total_loss += loss.detach()



# test dgmrf
# x = graphs["latent"].x
# Gx = dgmrf(graphs)
# prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) + dgmrf.log_det(len(graphs))) / N
#
# vi_entropy = 0
#
# obs_model = ObservationModel()
# y_hat = obs_model(x, graphs["latent", "observation", "data"])
# y = graphs["data"].x
#
# noise_std = graphs.get_example(0)['data'].noise_std
# log_noise_std = torch.log(noise_std)
# data_ll = -0.5 * torch.sum(torch.pow((y - y_hat), 2)) / (M * noise_std**2) - log_noise_std
#
# elbo = prior_ll + vi_entropy + data_ll
# print(prior_ll, data_ll)