import torch
import pytorch_lightning as pl
import numpy as np

class NodeMean(pl.LightningModule):

    def __init__(self, config, data, joint_mask, T=1, gt=None, **kwargs):
        super(NodeMean, self).__init__()

        self.T = T
        self.y = data
        self.mask = joint_mask
        self.inv_mask = torch.logical_not(self.mask).reshape(T, -1)
        self.y_masked = torch.zeros(self.mask.size(), dtype=self.y.dtype)
        self.y_masked[self.mask] = self.y

        if gt is not None:
            self.gt = gt

        self.node_means = torch.tensor(np.nanmean(self.y_masked.reshape(self.T, -1), 0))

    def test_step(self, test_index, *args):

        x_hat = self.y_masked.reshape(self.T, -1)
        x_hat[self.inv_mask] = self.node_means.unsqueeze(0).repeat(self.T, 1)[self.inv_mask]

        if hasattr(self, 'gt'):
            # use unobserved nodes to evaluate predictions
            # test_index indexes latent space
            test_mask = torch.logical_not(self.mask)
            gt_mean = self.gt[test_mask]
            residuals = (self.gt[test_mask] - x_hat[test_mask])
            rel_residuals = residuals / gt_mean

        else:
            # use held out part of data to evaluate predictions
            # test_index indexes data space
            masked_x = x_hat.flatten()[self.mask]
            data = self.y[test_index]
            residuals = (self.y[test_index] - masked_x[test_index])
            rel_residuals = residuals / data

        self.log("test_mae", residuals.abs().mean().item(), sync_dist=True)
        self.log("test_rmse", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)
        self.log("test_mse", torch.pow(residuals, 2).mean().item(), sync_dist=True)
        self.log("test_mape", rel_residuals.abs().mean().item(), sync_dist=True)



class KNN(pl.LightningModule):

    def __init__(self, config, data, joint_mask, spatial_graph, T=1, K=5, gt=None, **kwargs):
        super(NodeMean, self).__init__()

        self.T = T
        self.y = data
        self.mask = joint_mask
        self.inv_mask = torch.logical_not(self.mask).reshape(T, -1)
        self.y_masked = torch.zeros(self.mask.size(), dtype=self.y.dtype)
        self.y_masked[self.mask] = self.y

        if gt is not None:
            self.gt = gt

        self.pos = spatial_graph['pos']

        self.node_means = torch.tensor(np.nanmean(self.y_masked.reshape(self.T, -1), 0))

    def find_KNN(self):


    def test_step(self, test_index, *args):

        x_hat = self.y_masked.reshape(self.T, -1)
        x_hat[self.inv_mask] = self.node_means.unsqueeze(0).repeat(self.T, 1)[self.inv_mask]

        if hasattr(self, 'gt'):
            # use unobserved nodes to evaluate predictions
            # test_index indexes latent space
            test_mask = torch.logical_not(self.mask)
            gt_mean = self.gt[test_mask]
            residuals = (self.gt[test_mask] - x_hat[test_mask])
            rel_residuals = residuals / gt_mean

        else:
            # use held out part of data to evaluate predictions
            # test_index indexes data space
            masked_x = x_hat.flatten()[self.mask]
            data = self.y[test_index]
            residuals = (self.y[test_index] - masked_x[test_index])
            rel_residuals = residuals / data

        self.log("test_mae", residuals.abs().mean().item(), sync_dist=True)
        self.log("test_rmse", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)
        self.log("test_mse", torch.pow(residuals, 2).mean().item(), sync_dist=True)
        self.log("test_mape", rel_residuals.abs().mean().item(), sync_dist=True)



