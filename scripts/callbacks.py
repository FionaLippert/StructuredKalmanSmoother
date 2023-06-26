import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import torch_geometric as ptg
import numpy as np
from matplotlib import pyplot as plt

from structuredKS.models.dgmrf import *
from pems_plotting import *

class LogPredictionsCallback(Callback):

    def __init__(self, logger):
        self.logger = logger

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        for key, val in outputs.items():
            self.logger.log_image(key=key, images=[img for img in val])


class LatticeInferenceCallback(Callback):

    def __init__(self, logger, config, grid_size, true_post_mean, true_post_std, true_residuals):#, val_idx, train_idx):
        self.logger = logger
        self.config = config
        self.grid_size = grid_size
        self.true_post_mean = true_post_mean
        self.true_post_std = true_post_std
        self.true_residuals = true_residuals
        # self.val_idx = val_idx
        # self.train_idx = train_idx

    def masked_image(self, img, mask):
        return wandb.Image(img.cpu().detach().numpy(),
                           masks={'observations': {'mask_data': mask.cpu().detach().numpy(),
                                                   'class_labels': {0: "unobserved", 1: "observed"}}})

    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, 'gt'):
            img_shape = (pl_module.T, *self.grid_size)
            gt = pl_module.gt.reshape(img_shape)
            masks = pl_module.mask.reshape(img_shape)
            # self.logger.log_image(key='gt', images=[self.masked_image(gt[t], masks[t]) for t in range(pl_module.T)])
            self.logger.log_image(key='gt', images=[img for img in gt])
            self.logger.log_image(key='mask', images=[img.float() for img in masks])

    def on_train_epoch_start(self, trainer, pl_module):
        # plot current vi mean and std
        img_shape = (pl_module.T, *self.grid_size)
        mean, std = pl_module.vi_dist.posterior_estimate(pl_module.noise_var)
        mean = mean.reshape(img_shape)
        std = std.reshape(img_shape)

        self.logger.log_image(key='vi_mean', images=[img for img in mean])
        self.logger.log_image(key='vi_std', images=[img for img in std])

        if hasattr(pl_module, 'vi_dist_vx'):
            mean_vx = pl_module.vi_dist_vx.posterior_estimate()[0].reshape(*self.grid_size)
            mean_vy = pl_module.vi_dist_vy.posterior_estimate()[0].reshape(*self.grid_size)

            self.logger.log_image(key='vi_v_mean', images=[mean_vx, mean_vy])

    def on_test_end(self, trainer, pl_module):
        if hasattr(pl_module, 'gt'):
            img_shape = (pl_module.T, *self.grid_size)
            gt = pl_module.gt.reshape(img_shape).cpu().detach()

            # VI inference
            # vi_mean, vi_std = pl_module.vi_dist.posterior_estimate(pl_module.noise_var)
            vi_mean, vi_std = pl_module.vi_mean, pl_module.vi_std
            vi_mean = vi_mean.reshape(img_shape).cpu().detach()
            vi_std = vi_std.reshape(img_shape).cpu().detach()

            true_mean = self.true_post_mean.reshape(img_shape).cpu().detach()
            true_residuals = self.true_residuals.reshape(img_shape).cpu().detach()
            # true_std = self.true_post_std.reshape(img_shape).cpu().detach()

            # # CG inference
            # if self.config['use_hierarchy']:
            #     v_mean = torch.stack([pl_module.vi_dist_vx.mean_param, pl_module.vi_dist_vy.mean_param], dim=0).unsqueeze(1)
            #     print(v_mean.size())
            # else:
            #     v_mean = None
            # data = torch.zeros(pl_module.mask.size())
            # data[pl_module.mask] = pl_module.y
            # cg_mean = posterior_mean(pl_module.dgmrf, data.reshape(1, *pl_module.input_shape),
            #                            pl_module.mask, self.config, pl_module.noise_var,
            #                          initial_guess=vi_mean.reshape(1, -1, 1)).reshape(img_shape).cpu().detach()

            # cg_mean, cg_std = posterior_inference(pl_module.dgmrf, pl_module.y_masked.reshape(1, *pl_module.input_shape),
            #                                       pl_module.mask, self.config, pl_module.noise_var)
            post_mean, post_std = pl_module.post_mean, pl_module.post_std
            post_mean = post_mean.reshape(img_shape).cpu().detach()
            post_std = post_std.reshape(img_shape).cpu().detach()

            # cg_samples = sample_posterior(10, pl_module.dgmrf, pl_module.y_masked.reshape(1, *pl_module.input_shape),
            #                     pl_module.mask, self.config, pl_module.noise_var)
            # cg_samples = pl_module.cg_samples
            # cg_sample = cg_samples.mean(0).reshape(img_shape).cpu().detach()

            # cg_sample = sample_prior(1, pl_module.dgmrf, (1, pl_module.T, pl_module.num_nodes),
            #                          self.config).reshape(img_shape).cpu().detach()

            data = torch.ones_like(pl_module.mask) * np.nan
            data[pl_module.mask] = pl_module.y
            data = data.reshape(img_shape).cpu().detach()

            # residuals = gt - mean
            residuals = gt - post_mean
            # residuals_optimal = gt - true_mean

            vmin = gt.min()
            vmax = gt.max()

            fig, ax = plt.subplots(6, pl_module.T, figsize=(pl_module.T * 2, 6 * 2))
            for t in range(pl_module.T):
                # data
                ax[0, t].imshow(data[t], vmin=vmin, vmax=vmax)
                ax[0, t].set_title(f't = {t}', fontsize=10)
                ax[0, 0].set_ylabel('data', fontsize=10)

                # ground truth
                data_img = ax[1, t].imshow(gt[t], vmin=vmin, vmax=vmax)
                ax[1, 0].set_ylabel('ground truth', fontsize=10)

                # true posterior mean
                # ax[2, t].imshow(true_mean[t], vmin=vmin, vmax=vmax)
                # ax[2, 0].set_ylabel('true posterior mean', fontsize=30)
                ax[2, t].imshow(post_mean[t], vmin=vmin, vmax=vmax)
                ax[2, 0].set_ylabel('posterior mean', fontsize=10)

                # VI posterior mean
                ax[3, t].imshow(vi_mean[t])
                ax[3, 0].set_ylabel('VI posterior mean', fontsize=10)
                # ax[3, t].imshow(mean[t], vmin=vmin, vmax=vmax)
                # ax[3, 0].set_ylabel('VI mean', fontsize=30)
                # res_img_opt = ax[3, t].imshow(true_residuals[t], vmin=-true_residuals.abs().max(), vmax=true_residuals.abs().max(),
                #                           cmap='coolwarm')
                # ax[3, 0].set_ylabel('residuals optimal', fontsize=30)

                # VI posterior std
                std_img = ax[4, t].imshow(post_std[t], vmin=post_std.min(), vmax=post_std.max(), cmap='Reds')
                ax[4, 0].set_ylabel('posterior std', fontsize=10)

                # residuals
                res_img = ax[5, t].imshow(residuals[t], vmin=-residuals.abs().max(), vmax=residuals.abs().max(),
                                          cmap='coolwarm')
                ax[5, 0].set_ylabel('residuals', fontsize=10)

            cbar1 = fig.colorbar(data_img, ax=ax[:4, :], shrink=0.6, aspect=10)
            cbar2 = fig.colorbar(std_img, ax=ax[4, :], shrink=0.8, aspect=4)
            cbar3 = fig.colorbar(res_img, ax=ax[5, :], shrink=0.8, aspect=4)

            # cbar1 = fig.colorbar(data_img, ax=ax[:3, :], shrink=0.6, aspect=10)
            # cbar2a = fig.colorbar(res_img_opt, ax=ax[3, :], shrink=0.8, aspect=4)
            # cbar2 = fig.colorbar(std_img, ax=ax[4, :], shrink=0.8, aspect=4)
            # cbar3 = fig.colorbar(res_img, ax=ax[5, :], shrink=0.8, aspect=4)

            cbar1.ax.tick_params(labelsize=10)
            cbar2.ax.tick_params(labelsize=10)
            # cbar2a.ax.tick_params(labelsize=20)
            cbar3.ax.tick_params(labelsize=10)

            wandb.log({'overview': wandb.Image(plt)})
            plt.close(fig)


            # # plot example timeseries
            # mask = pl_module.mask #.reshape(img_shape).cpu().detach()
            # data = data.reshape(pl_module.T, -1)
            # data[data == 0] = np.nan
            # vi_mean = vi_mean.reshape(pl_module.T, -1).cpu()
            # true_mean = true_mean.reshape(pl_module.T, -1).cpu()
            # fig, ax = plt.subplots(figsize=(10, 6))
            # l = ax.plot(vi_mean.mean(1), label='VI mean')
            # ax.fill_between(range(pl_module.T), vi_mean.mean(1) - vi_mean.std(1), vi_mean.mean(1) + vi_mean.std(1),
            #                 alpha=0.2, color=l[0].get_color())
            # l = ax.plot(true_mean.mean(1), label='CG mean')
            # ax.fill_between(range(pl_module.T), true_mean.mean(1) - true_mean.std(1),
            #                 true_mean.mean(1) + true_mean.std(1),
            #                 alpha=0.2, color=l[0].get_color())
            # data_mean = np.nanmean(data.cpu().numpy(), axis=1)
            # data_std = np.nanstd(data.cpu().numpy(), axis=1)
            # l = ax.plot(data_mean, label='data mean')
            # ax.fill_between(range(pl_module.T), data_mean - data_std, data_mean + data_std,
            #                 alpha=0.2, color=l[0].get_color())
            # ax.set(xlabel='time')
            # ax.legend()
            # self.logger.log_image(key='mean_timeseries', images=[fig])
            # plt.close()

class GraphInferenceCallback(Callback):

    def __init__(self, logger, config, graph_t, tidx, test_nodes, val_nodes, subset=None, mark_subset=None):
        self.logger = logger
        self.config = config
        self.graph_t = graph_t
        self.tidx = tidx
        self.subset = subset
        self.mark_subset = mark_subset

        pos = graph_t.pos.cpu()

        if subset is not None:
            # self.train_nodes = [i for i in train_nodes if i in subset]
            # self.val_nodes = [i for i in val_nodes if i in subset]
            # self.train_val_nodes = [i for i in subset if i not in test_nodes]
            self.test_nodes = [i for i in test_nodes if i in subset]
            self.val_nodes = [i for i in val_nodes if i in subset]
            self.all_nodes = subset.cpu()

            sub_edges, sub_attr = ptg.utils.subgraph(self.subset, self.graph_t.edge_index,
                                              edge_attr=self.graph_t.edge_weight, relabel_nodes=True)
            self.subgraph = ptg.data.Data(edge_index=sub_edges, num_nodes=len(self.subset), edge_attr=sub_attr)

            fig, ax = plt.subplots(figsize=(15, 10))

            ax.scatter(pos[self.all_nodes, 0], pos[self.all_nodes, 1], alpha=0.1, color='gray', s=10)
            # ax.scatter(pos[self.train_nodes, 0], pos[self.train_nodes, 1], alpha=0.5, color='red', s=10)
            ax.scatter(pos[self.val_nodes, 0], pos[self.val_nodes, 1], alpha=0.5, color='orange', s=10)
            ax.scatter(pos[self.test_nodes, 0], pos[self.test_nodes, 1], alpha=0.5, color='green', s=10)
            # for idx in self.train_nodes:
            #     ax.text(pos[idx, 0], pos[idx, 1], str(int(idx)), fontsize=8, color='red')
            for idx in self.val_nodes:
                ax.text(pos[idx, 0], pos[idx, 1], str(int(idx)), fontsize=8, color='orange')
            for idx in self.test_nodes:
                ax.text(pos[idx, 0], pos[idx, 1], str(int(idx)), fontsize=8, color='green')

            indices = np.arange(len(self.subset))
            sub_pos = self.graph_t.pos[self.subset].cpu().numpy()
            plot_nodes(dir, self.subgraph, sub_pos, np.ones(len(self.subset)), indices, fig, ax,
                       node_alpha=0, edge_alpha=0.8, edge_width=sub_attr/sub_attr.max())

            self.logger.log_image(key=f'subset_nodes', images=[fig])
        else:
            self.val_nodes = val_nodes
            # self.train_nodes = train_nodes
            self.test_nodes = test_nodes
            self.all_nodes = torch.arange(pos.size(0)).cpu()
            # self.train_val_nodes = [i for i in self.all_nodes if not i in test_nodes]



        fig, ax = plt.subplots(figsize=(15, 10))
        ax.scatter(pos[self.all_nodes, 0], pos[self.all_nodes, 1], alpha=0.1, color='gray', s=2)
        # ax.scatter(pos[self.train_nodes, 0], pos[self.train_nodes, 1], alpha=0.5, color='red', s=8)
        ax.scatter(pos[self.val_nodes, 0], pos[self.val_nodes, 1], alpha=0.5, color='orange', s=8)
        ax.scatter(pos[self.test_nodes, 0], pos[self.test_nodes, 1], alpha=0.5, color='green', s=8)
        # for idx in self.train_nodes:
        #     ax.text(pos[idx, 0], pos[idx, 1], str(int(idx)), fontsize=8, color='red')
        for idx in self.val_nodes:
            ax.text(pos[idx, 0], pos[idx, 1], str(int(idx)), fontsize=8, color='orange')
        for idx in self.test_nodes:
            ax.text(pos[idx, 0], pos[idx, 1], str(int(idx)), fontsize=8, color='green')
        self.logger.log_image(key='test_node_map', images=[fig])




    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        # vi_mean = pl_module.vi_dist.mean_param.reshape(pl_module.T, -1)
        vi_mean, vi_std = pl_module.vi_dist.posterior_estimate(pl_module.noise_var)
        vi_mean = vi_mean.reshape(pl_module.T, -1).cpu()
        vi_std = vi_std.reshape(pl_module.T, -1).cpu()
        data = pl_module.y_masked.reshape(pl_module.T, -1).cpu()
        data[data == 0] = np.nan

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in self.val_nodes: #[:3]:
            l = ax.plot(vi_mean[:, idx], label=f'node {idx}')
            ax.fill_between(range(pl_module.T), vi_mean[:, idx] - vi_std[:, idx], vi_mean[:, idx] + vi_std[:, idx],
                            alpha=0.2, color=l[0].get_color())
            # for jdx in range(vi_samples.size(0)):
            #     ax.plot(vi_samples[jdx, :, idx].cpu(), c=l[0].get_color(), alpha=0.1)
            ax.plot(data[:, idx], '--', c=l[0].get_color(), alpha=0.6)
        ax.legend()
        self.logger.log_image(key='example_val_timeseries', images=[fig])
        plt.close()

        # fig, ax = plt.subplots(figsize=(10, 6))
        # for idx in self.train_nodes: #[:3]:
        #     l = ax.plot(vi_mean[:, idx], label=f'node {idx}')
        #     ax.fill_between(range(pl_module.T), vi_mean[:, idx] - vi_std[:, idx], vi_mean[:, idx] + vi_std[:, idx],
        #                     alpha=0.2, color=l[0].get_color())
        #     # for jdx in range(vi_samples.size(0)):
        #     #     ax.plot(vi_samples[jdx, :, idx].cpu(), c=l[0].get_color(), alpha=0.1)
        #     ax.plot(data[:, idx], '--', c=l[0].get_color(), alpha=0.6)
        # ax.legend()
        # self.logger.log_image(key='example_train_timeseries', images=[fig])
        # plt.close()

    def on_train_start(self, trainer, pl_module):

        fig, ax = plt.subplots(figsize=(15, 10))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")

        indices = pl_module.mask.reshape(pl_module.T, -1)[self.tidx].nonzero().squeeze().cpu().numpy()
        pos = self.graph_t.pos.cpu().numpy()
        data = pl_module.y_masked.reshape(pl_module.T, -1)[self.tidx, indices].cpu().numpy()
        plot_nodes(dir, self.graph_t, pos, data,
                   indices, fig, ax, cax)
        self.logger.log_image(key=f'data_tidx={self.tidx}', images=[fig])
        plt.close()


    def on_test_end(self, trainer, pl_module):

        # vi_mean, vi_std = pl_module.vi_dist.posterior_estimate(pl_module.noise_var)

        post_mean, post_std = pl_module.post_mean, pl_module.post_std
        # vi_mean, vi_std = pl_module.vi_mean, pl_module.vi_std
        mean = post_mean.reshape(pl_module.T, -1)[self.tidx].detach()
        std = post_std.reshape(pl_module.T, -1)[self.tidx].detach()
        indices = np.arange(self.graph_t.num_nodes)
        pos = self.graph_t.pos.cpu().numpy()

        # plot vi mean
        fig, ax = plt.subplots(figsize=(15, 10))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        plot_nodes(dir, self.graph_t, pos, mean.cpu().numpy(),
                   indices, fig, ax, cax)
        self.logger.log_image(key=f'post_mean_tidx={self.tidx}', images=[fig])
        plt.close()

        # plot vi std
        fig, ax = plt.subplots(figsize=(15, 10))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        plot_nodes(dir, self.graph_t, pos, std.cpu().numpy(),
                   indices, fig, ax, cax)
        self.logger.log_image(key=f'post_std_tidx={self.tidx}', images=[fig])
        plt.close()

        if self.subset is not None:
            print('plot subset')

            indices = np.arange(len(self.subset))
            pos = self.graph_t.pos[self.subset].cpu().numpy()

            fig, ax = plt.subplots(figsize=(15, 10))
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="7%", pad="2%")

            plot_nodes(dir, self.subgraph, pos, mean[self.subset].cpu().numpy(), indices, fig, ax, cax,
                       mark_indices=self.mark_subset.cpu().numpy())
            self.logger.log_image(key=f'subset_post_mean_tidx={self.tidx}', images=[fig])
            plt.close()

            fig, ax = plt.subplots(figsize=(15, 10))
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="7%", pad="2%")

            plot_nodes(dir, self.subgraph, pos, std[self.subset].cpu().numpy(), indices, fig, ax, cax,
                       mark_indices=self.mark_subset.cpu().numpy())
            self.logger.log_image(key=f'subset_post_std_tidx={self.tidx}', images=[fig])
            plt.close()

        # # use CG to approximate true posterior
        # data = torch.zeros(pl_module.mask.size(), dtype=pl_module.y.dtype)
        # data[pl_module.mask] = pl_module.y
        # true_mean = posterior_mean(pl_module.dgmrf, data.reshape(1, *pl_module.input_shape),
        #                            pl_module.mask, self.config, v=vi_mean).detach() #.cpu().detach()
        #
        # # plot CG mean
        # fig, ax = plt.subplots(figsize=(15, 10))
        # ax_divider = make_axes_locatable(ax)
        # cax = ax_divider.append_axes("right", size="7%", pad="2%")
        # plot_nodes(dir, self.graph_t, pos, true_mean.reshape(pl_module.T, -1)[self.tidx].cpu().numpy(),
        #            indices, fig, ax, cax)
        # self.logger.log_image(key=f'CG_mean_tidx={self.tidx}', images=[fig])
        # plt.close()

        post_mean = post_mean.reshape(pl_module.T, -1).cpu()
        post_std = post_std.reshape(pl_module.T, -1).cpu()
        data = pl_module.y_masked.reshape(pl_module.T, -1).cpu()
        data[data == 0] = np.nan

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in self.test_nodes:  # [:3]:
            l = ax.plot(post_mean[:, idx], label=f'node {idx}')
            ax.fill_between(range(pl_module.T), post_mean[:, idx] - post_std[:, idx],
                            post_mean[:, idx] + post_std[:, idx],
                            alpha=0.2, color=l[0].get_color())
            # for jdx in range(vi_samples.size(0)):
            #     ax.plot(vi_samples[jdx, :, idx].cpu(), c=l[0].get_color(), alpha=0.1)
            ax.plot(data[:, idx], '--', c=l[0].get_color(), alpha=0.6)
        ax.legend()
        self.logger.log_image(key='example_test_timeseries', images=[fig])
        plt.close()

        # plot mean difference between VI and CG mean
        data = pl_module.y_masked.reshape(pl_module.T, -1)
        data[data == 0] = np.nan
        # vi_mean = vi_mean.reshape(pl_module.T, -1).cpu()
        # true_mean = true_mean.reshape(pl_module.T, -1).cpu()
        fig, ax = plt.subplots(figsize=(10, 6))
        l = ax.plot(post_mean.mean(1), label='CG mean')
        ax.fill_between(range(pl_module.T), post_mean.mean(1)-post_mean.std(1), post_mean.mean(1)+post_mean.std(1),
                        alpha=0.2, color=l[0].get_color())
        # l = ax.plot(true_mean.mean(1), label='CG mean')
        # ax.fill_between(range(pl_module.T), true_mean.mean(1) - true_mean.std(1), true_mean.mean(1) + true_mean.std(1),
        #                 alpha=0.2, color=l[0].get_color())
        data_mean = np.nanmean(data.cpu().numpy(), axis=1)
        data_std = np.nanstd(data.cpu().numpy(), axis=1)
        l = ax.plot(data_mean, label='data mean')
        ax.fill_between(range(pl_module.T), data_mean - data_std, data_mean + data_std,
                        alpha=0.2, color=l[0].get_color())
        ax.set(xlabel='time')
        ax.legend()
        self.logger.log_image(key='mean_timeseries', images=[fig])
        plt.close()

