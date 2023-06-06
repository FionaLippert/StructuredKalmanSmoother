import torch
import pytorch_lightning as pl
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from structuredKS.utils import crps_score, int_score
import os
import pickle

class NodeARIMA(pl.LightningModule):

    def __init__(self, config, data, joint_mask, T=1, gt=None, **kwargs):
        super(NodeARIMA, self).__init__()

        # (p, 0, 0) = AR(p), (0, 0, q) = MA(q), (p, d, q) = ARIMA
        self.arima_params = config.get('arima_params', (1, 0, 0))
        self.T = T
        self.y = data
        self.mask = joint_mask
        self.inv_mask = torch.logical_not(self.mask).reshape(T, -1)
        self.N = self.inv_mask.size(1)
        self.y_masked = torch.ones(self.mask.size(), dtype=self.y.dtype) * np.nan
        self.y_masked[self.mask] = self.y

        if gt is not None:
            self.gt = gt

        self.node_means = torch.tensor(np.nanmean(self.y_masked.reshape(self.T, -1), 0))

    def test_step(self, test_index, *args):

        if hasattr(self, 'gt'):
            # use unobserved nodes to evaluate predictions
            # test_index indexes latent space
            test_mask = torch.logical_not(self.mask)
            y_masked = torch.ones_like(self.y_masked) * np.nan
            y_masked[self.mask] = self.y_masked[self.mask]
            y_masked = y_masked.reshape(self.T, -1)

        else:
            test_mask = self.mask.nonzero()[predict_index]  # test data
            data_mask = self.mask.nonzero()[~predict_index]  # all data except test data
            y_masked = torch.ones_like(self.y_masked) * np.nan
            y_masked[data_mask] = self.y_masked[data_mask]
            y_masked = y_masked.reshape(self.T, -1)


        x_hat = np.zeros((self.T, self.N))
        x_smoothed = np.zeros((self.T, self.N))
        std = np.zeros((self.T, self.N))
        std_smoothed = np.zeros((self.T, self.N))
        for i in range(self.N):
            d_i = y_masked[:, i].detach().numpy()
            mean_i = np.nanmean(d_i)
            model = SARIMAX(d_i - mean_i, order=self.arima_params,
                          trend=None, enforce_stationarity=False).fit()
            prediction = model.get_prediction(0, self.T-1)
            x_hat[:, i] = prediction.predicted_mean + mean_i
            std[:, i] = prediction.se_mean

            x_smoothed[:, i] = model.states.smoothed[:, 0] + mean_i
            std_smoothed[:, i] = model.states.smoothed_cov[:, 0, 0]


        gt_mean = self.gt[test_mask].detach().numpy()
        x_hat = x_hat.flatten()[test_mask.detach().numpy()]
        std = std.flatten()[test_mask.detach().numpy()]
        x_smoothed = x_smoothed.flatten()[test_mask.detach().numpy()]
        std_smoothed = std_smoothed.flatten()[test_mask.detach().numpy()]

        residuals = (gt_mean - x_hat)
        rel_residuals = residuals / gt_mean

        residuals_s = (gt_mean - x_smoothed)
        rel_residuals_s = residuals_s / gt_mean


        self.log("test_crps", crps_score(x_hat, std, gt_mean), sync_dist=True)
        self.log("test_int_score", int_score(x_hat, std, gt_mean), sync_dist=True)

        self.log("test_mae", np.abs(residuals).mean(), sync_dist=True)
        self.log("test_rmse", np.sqrt(np.square(residuals).mean()), sync_dist=True)
        self.log("test_mse", np.square(residuals).mean(), sync_dist=True)
        self.log("test_mape", np.abs(rel_residuals).mean(), sync_dist=True)

        self.log("test_crps_smoothed", crps_score(x_smoothed, std_smoothed, gt_mean), sync_dist=True)
        self.log("test_int_score_smoothed", int_score(x_smoothed, std_smoothed, gt_mean), sync_dist=True)
        self.log("test_mae_smoothed", np.abs(residuals_s).mean(), sync_dist=True)
        self.log("test_rmse_smoothed", np.sqrt(np.square(residuals_s).mean()), sync_dist=True)
        self.log("test_mse_smoothed", np.square(residuals_s).mean(), sync_dist=True)
        self.log("test_mape_smoothed", np.abs(rel_residuals_s).mean(), sync_dist=True)

    def predict_step(self, predict_index, *args):


        if hasattr(self, 'gt'):
            # use unobserved nodes to evaluate predictions
            # test_index indexes latent space
            test_mask = torch.logical_not(self.mask)
            y_masked = torch.ones_like(self.y_masked) * np.nan
            y_masked[self.mask] = self.y_masked[self.mask]
            y_masked = y_masked.reshape(self.T, -1)

        else:
            test_mask = self.mask.nonzero()[predict_index] # test data
            data_mask = self.mask.nonzero()[~predict_index]  # all data except test data
            y_masked = torch.ones_like(self.y_masked) * np.nan
            y_masked[data_mask] = self.y_masked[data_mask]
            y_masked = y_masked.reshape(self.T, -1)

        x_hat = np.zeros((self.T, self.N))
        std = np.zeros((self.T, self.N))
        x_smoothed = np.zeros((self.T, self.N))
        std_smoothed = np.zeros((self.T, self.N))
        for i in range(self.N):
            d_i = y_masked[:, i].detach().numpy()
            mean_i = np.nanmean(d_i)
            model = SARIMAX(d_i - mean_i, order=self.arima_params,
                          trend=None, enforce_stationarity=False).fit()
            prediction = model.get_prediction(0, self.T-1)
            x_hat[:, i] = prediction.predicted_mean + mean_i
            std[:, i] = prediction.se_mean

            x_smoothed[:, i] = model.states.smoothed[:, 0] + mean_i
            std_smoothed[:, i] = model.states.smoothed_cov[:, 0, 0]


        return {'pred_mean': x_hat.reshape(self.T, self.N),
                'pred_std': std.reshape(self.T, self.N),
                'post_mean': x_smoothed.reshape(self.T, self.N),
                'post_std': std_smoothed.reshape(self.T, self.N),
                'data': y_masked,
                'gt': self.gt if hasattr(self, 'gt') else 'NA'}



#
# class IndependentKF(pl.LightningModule):
#
#     def __init__(self, config, data, joint_mask, T=1, gt=None, **kwargs):
#         super(IndependentKF, self).__init__()
#
#         self.T = T
#         self.y = data
#         self.mask = joint_mask.reshape(T, -1)
#         self.N = self.mask.size(1)
#         self.inv_mask = torch.logical_not(self.mask).reshape(T, -1)
#         self.y_masked = torch.zeros(self.mask.size(), dtype=self.y.dtype)
#         self.y_masked[self.mask] = self.y
#
#         if gt is not None:
#             self.gt = gt
#
#         self.mean_0 = torch.nn.parameter.Paramter(torch.zeros(1,))
#         self.std_0 = torch.nn.parameter.Paramter(torch.rand(1, ))
#         self.transition_param = torch.nn.parameter.Paramter(torch.ones(1,))
#         self.std_t = torch.nn.parameter.Paramter(torch.rand(1, ))
#
#         if config["learn_noise_std"]:
#             self.obs_std = torch.nn.parameter.Parameter(torch.tensor(config["noise_std"]))
#         else:
#             self.obs_std = torch.tensor(config["noise_std"])
#
#     def prediction_step(self, state, var):
#
#         return self.transition_param * state, var * self.transition_param**2 + self.std_t**2
#
#     def filter_step(self, state, var, t):
#
#         K = var * self.mask[t] * (1. / (var + self.obs_std**2))
#
#         new_state = state + K * (self.y_masked[t] - state)
#         new_var = (1 - K) * var
#
#         return new_state, new_var
#
#     def training_step(self, training_idx):
#
#         self.train_mask = torch.zeros_like(self.mask)
#         self.train_mask[self.mask.nonzero()[training_index.squeeze()]] = 1
#
#
#




