import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import numpy as np
from matplotlib import pyplot as plt

from structuredKS.models.dgmrf import posterior_mean
from pems_plotting import *

class LogPredictionsCallback(Callback):

    def __init__(self, logger):
        self.logger = logger

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        for key, val in outputs.items():
            self.logger.log_image(key=key, images=[img for img in val])


class LatticeInferenceCallback(Callback):

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

    def masked_image(self, img, mask):
        return wandb.Image(img.cpu().detach().numpy(),
                           masks={'observations': {'mask_data': mask.cpu().detach().numpy(),
                                                   'class_labels': {0: "unobserved", 1: "observed"}}})

    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, 'gt'):
            grid_size = int(np.sqrt(pl_module.num_nodes))
            img_shape = (pl_module.T, grid_size, grid_size)
            gt = pl_module.gt.reshape(img_shape)
            masks = pl_module.mask.reshape(img_shape)
            # self.logger.log_image(key='gt', images=[self.masked_image(gt[t], masks[t]) for t in range(pl_module.T)])
            self.logger.log_image(key='gt', images=[img for img in gt])
            self.logger.log_image(key='mask', images=[img.float() for img in masks])

    def on_train_epoch_start(self, trainer, pl_module):
        # plot current vi mean and std
        grid_size = int(np.sqrt(pl_module.num_nodes))
        img_shape = (pl_module.T, grid_size, grid_size)
        mean, std = pl_module.vi_dist.posterior_estimate()
        mean = mean.reshape(img_shape)
        std = std.reshape(img_shape)

        self.logger.log_image(key='vi_mean', images=[img for img in mean])
        self.logger.log_image(key='vi_std', images=[img for img in std])

        if hasattr(pl_module, 'vi_dist_vx'):
            mean_vx = pl_module.vi_dist_vx.posterior_estimate()[0].reshape(grid_size, grid_size)
            mean_vy = pl_module.vi_dist_vy.posterior_estimate()[0].reshape(grid_size, grid_size)

            self.logger.log_image(key='vi_v_mean', images=[mean_vx, mean_vy])

    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, 'gt'):
            grid_size = int(np.sqrt(pl_module.num_nodes))
            img_shape = (pl_module.T, grid_size, grid_size)
            gt = pl_module.gt.reshape(img_shape).cpu().detach()

            # VI inference
            mean, std = pl_module.vi_dist.posterior_estimate()
            mean = mean.reshape(img_shape).cpu().detach()
            std = std.reshape(img_shape).cpu().detach()

            # CG inference
            if self.config['use_hierarchy']:
                v_mean = torch.stack([pl_module.vi_dist_vx.mean_param, pl_module.vi_dist_vy.mean_param], dim=0).unsqueeze(1)
                print(v_mean.size())
            else:
                v_mean = None
            data = torch.zeros(pl_module.mask.size())
            data[pl_module.mask] = pl_module.y
            true_mean = posterior_mean(pl_module.dgmrf, data.reshape(1, *pl_module.input_shape),
                                       pl_module.mask, self.config, v=v_mean).reshape(img_shape).cpu().detach()

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

class GraphInferenceCallback(Callback):

    def __init__(self, logger, config, graph_t, tidx, val_nodes, train_nodes):
        self.logger = logger
        self.config = config
        self.val_nodes = val_nodes
        self.train_nodes = train_nodes
        self.graph_t = graph_t
        self.tidx = tidx

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        vi_mean = pl_module.vi_dist.mean_param.reshape(pl_module.T, -1)
        data = pl_module.y_masked.reshape(pl_module.T, -1)

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in self.val_nodes[:3]:
            l = ax.plot(vi_mean[:, idx].cpu(), label=f'node {idx}')
            ax.plot(data[:, idx].cpu(), c=l[0].get_color(), ls='--')
        ax.legend()
        self.logger.log_image(key='example_val_timeseries', images=[fig])
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in self.train_nodes[:3]:
            l = ax.plot(vi_mean[:, idx].cpu(), label=f'node {idx}')
            ax.plot(data[:, idx].cpu(), c=l[0].get_color(), ls='--')
        ax.legend()
        self.logger.log_image(key='example_train_timeseries', images=[fig])
        plt.close()

    def on_train_start(self, trainer, pl_module):

        fig, ax = plt.subplots(figsize=(15, 10))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")

        indices = self.graph_t.mask.nonzero().squeeze().cpu().numpy()
        pos = self.graph_t.pos.cpu().numpy()
        data = pl_module.y_masked.reshape(pl_module.T, -1)[self.tidx, indices].cpu().numpy()
        plot_nodes(dir, self.graph_t, pos, data,
                   indices, ax, cax, fig)
        self.logger.log_image(key=f'data_tidx={self.tidx}', images=[fig])
        plt.close()

    def on_train_end(self, trainer, pl_module):

        vi_mean, vi_std = pl_module.vi_dist.posterior_estimate()
        mean = vi_mean.reshape(pl_module.T, -1)[self.tidx].detach()
        std = vi_std.reshape(pl_module.T, -1)[self.tidx].detach()
        indices = np.arange(self.graph_t.num_nodes)
        pos = self.graph_t.pos.cpu().numpy()

        # plot vi mean
        fig, ax = plt.subplots(figsize=(15, 10))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        plot_nodes(dir, self.graph_t, pos, mean.cpu().numpy(),
                   indices, ax, cax, fig)
        self.logger.log_image(key=f'vi_mean_tidx={self.tidx}', images=[fig])
        plt.close()

        # plot vi std
        fig, ax = plt.subplots(figsize=(15, 10))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        plot_nodes(dir, self.graph_t, pos, std.cpu().numpy(),
                   indices, ax, cax, fig)
        self.logger.log_image(key=f'vi_std_tidx={self.tidx}', images=[fig])
        plt.close()

        # use CG to approximate true posterior
        data = torch.zeros(pl_module.mask.size(), dtype=pl_module.y.dtype)
        data[pl_module.mask] = pl_module.y
        true_mean = posterior_mean(pl_module.dgmrf, data.reshape(1, *pl_module.input_shape),
                                   pl_module.mask, self.config, v=vi_mean).detach() #.cpu().detach()

        # plot CG mean
        fig, ax = plt.subplots(figsize=(15, 10))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        plot_nodes(dir, self.graph_t, pos, true_mean.reshape(pl_module.T, -1)[self.tidx].cpu().numpy(),
                   indices, ax, cax, fig)
        self.logger.log_image(key=f'CG_mean_tidx={self.tidx}', images=[fig])
        plt.close()

        # plot mean difference between VI and CG mean
        data = data.reshape(pl_module.T, -1)
        data[data == 0] = np.nan
        vi_mean = vi_mean.reshape(pl_module.T, -1).cpu()
        true_mean = true_mean.reshape(pl_module.T, -1).cpu()
        fig, ax = plt.subplots(figsize=(10, 6))
        l = ax.plot(vi_mean.mean(1), label='VI mean')
        ax.fill_between(range(pl_module.T), vi_mean.mean(1)-vi_mean.std(1), vi_mean.mean(1)+vi_mean.std(1),
                        alpha=0.2, color=l[0].get_color())
        l = ax.plot(true_mean.mean(1), label='CG mean')
        ax.fill_between(range(pl_module.T), true_mean.mean(1) - true_mean.std(1), true_mean.mean(1) + true_mean.std(1),
                        alpha=0.2, color=l[0].get_color())
        data_mean = np.nanmean(data.cpu().numpy(), axis=1)
        data_std = np.nanstd(data.cpu().numpy(), axis=1)
        l = ax.plot(data_mean, label='data mean')
        ax.fill_between(range(pl_module.T), data_mean - data_std, data_mean + data_std,
                        alpha=0.2, color=l[0].get_color())
        ax.set(xlabel='time')
        ax.legend()
        self.logger.log_image(key='mean_timeseries', images=[fig])
        plt.close()

