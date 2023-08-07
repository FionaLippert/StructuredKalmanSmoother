import torch
import pytorch_lightning as pl
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import BayesianRidge
from structuredKS.utils import crps_score, int_score
import os
import pickle

class NodeARIMA(pl.LightningModule):

    def __init__(self, config, data, joint_mask, train_mask, T=1, gt=None, **kwargs):
        super(NodeARIMA, self).__init__()

        # (p, 0, 0) = AR(p), (0, 0, q) = MA(q), (p, d, q) = ARIMA
        self.arima_params = config.get('arima_params', (1, 0, 0))
        self.seasonal_params = config.get('seasonal_params', (0, 0, 0, 0))
        self.T = T
        self.y = data
        self.mask = joint_mask
        self.inv_mask = torch.logical_not(self.mask).reshape(T, -1)
        self.N = self.inv_mask.size(1)
        self.y_masked = torch.ones(self.mask.size(), dtype=self.y.dtype) * np.nan
        self.y_masked[self.mask] = self.y

        self.train_mask = train_mask
        self.final_prediction = False # set to True for final model fitting and prediction

        if gt is not None:
            self.gt = gt
        else:
            self.gt = self.y_masked

        self.node_means = torch.tensor(np.nanmean(self.y_masked.reshape(self.T, -1), 0))

    def forward(self, data_mask, nodes='all'):

        y_masked = torch.ones_like(self.y_masked) * np.nan
        y_masked[data_mask] = self.y_masked[data_mask]
        y_masked = y_masked.reshape(self.T, -1)

        if nodes == 'all':
            nodes = torch.arange(self.N)

        x_hat = np.zeros((self.T, self.N))
        std = np.zeros((self.T, self.N))
        x_smoothed = np.zeros((self.T, self.N))
        std_smoothed = np.zeros((self.T, self.N))
        bic = np.ones(self.N) * np.nan
        for i in nodes:
            d_i = y_masked[:, i].detach().numpy()
            mean_i = np.nanmean(d_i)
            model = SARIMAX(d_i - mean_i, order=self.arima_params,
                            trend=None, enforce_stationarity=True, seasonal_order=self.seasonal_params).fit()
            print(model.summary())
            bic[i] = model.bic

            prediction = model.get_prediction(0, self.T - 1)
            x_hat[:, i] = prediction.predicted_mean + mean_i
            std[:, i] = prediction.se_mean

            x_smoothed[:, i] = model.states.smoothed[:, 0] + mean_i
            std_smoothed[:, i] = model.states.smoothed_cov[:, 0, 0]

        return {'pred_mean': x_hat.reshape(self.T, self.N),
                'pred_std': std.reshape(self.T, self.N),
                'post_mean': x_smoothed.reshape(self.T, self.N),
                'post_std': std_smoothed.reshape(self.T, self.N),
                'data': y_masked,
                'train_mask': data_mask,
                'gt': self.gt if hasattr(self, 'gt') else 'NA',
                'bic': bic}

    def evaluate(self, results, mask, split='test'):

        mask = mask.to(self.gt.device)

        gt_mean = self.gt[mask].detach().numpy()
        x_hat = results['pred_mean'].flatten()[mask.detach().numpy()]
        std = results['pred_std'].flatten()[mask.detach().numpy()]
        x_smoothed = results['post_mean'].flatten()[mask.detach().numpy()]
        std_smoothed = results['post_std'].flatten()[mask.detach().numpy()]

        residuals = (gt_mean - x_hat)
        rel_residuals = residuals / gt_mean

        residuals_s = (gt_mean - x_smoothed)
        rel_residuals_s = residuals_s / gt_mean

        self.log(f"{split}_bic", np.nanmean(results['bic']), sync_dist=True)

        self.log(f"{split}_crps", crps_score(x_hat, std, gt_mean), sync_dist=True)
        self.log(f"{split}_int_score", int_score(x_hat, std, gt_mean), sync_dist=True)

        self.log(f"{split}_mae", np.abs(residuals).mean(), sync_dist=True)
        self.log(f"{split}_rmse", np.sqrt(np.square(residuals).mean()), sync_dist=True)
        self.log(f"{split}_mse", np.square(residuals).mean(), sync_dist=True)
        self.log(f"{split}_mape", np.abs(rel_residuals).mean(), sync_dist=True)

        self.log(f"{split}_crps_smoothed", crps_score(x_smoothed, std_smoothed, gt_mean), sync_dist=True)
        self.log(f"{split}_int_score_smoothed", int_score(x_smoothed, std_smoothed, gt_mean), sync_dist=True)
        self.log(f"{split}_mae_smoothed", np.abs(residuals_s).mean(), sync_dist=True)
        self.log(f"{split}_rmse_smoothed", np.sqrt(np.square(residuals_s).mean()), sync_dist=True)
        self.log(f"{split}_mse_smoothed", np.square(residuals_s).mean(), sync_dist=True)
        self.log(f"{split}_mape_smoothed", np.abs(rel_residuals_s).mean(), sync_dist=True)


    def test_step(self, test_mask, *args):
        if self.final_prediction:
            # use all data except for test data
            # data_mask = torch.logical_and(self.mask, torch.logical_not(test_mask).flatten())
            print(self.mask.device, test_mask.device)
            data_mask = torch.logical_and(self.mask, torch.logical_not(test_mask.flatten()).to(self.mask.device))
            split = 'test'
        else:
            # only use dedicated training data (neither validation nor test data)
            data_mask = self.train_mask
            split = 'val'

        test_nodes = test_mask.reshape(self.T, -1).sum(0).nonzero().flatten()

        results = self.forward(data_mask, test_nodes)
        self.evaluate(results, test_mask.flatten(), split=split)


    def predict_step(self, predict_mask, *args):

        if self.final_prediction:
            # use all data except for test data
            # data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask).flatten())
            data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask.flatten()).to(self.mask.device))
        else:
            # only use dedicated training data (neither validation nor test data)
            data_mask = self.train_mask

        predict_nodes = predict_mask.reshape(self.T, -1).sum(0).nonzero().flatten()
        results = self.forward(data_mask, predict_nodes)

        return results


class MLP(pl.LightningModule):

    def __init__(self, config, targets, features, **kwargs):
        super(MLP, self).__init__()

        self.targets = targets # shape [T * num_nodes]
        self.features = features # shape [T * num_nodes, num_features]

        n_hidden = config.get('MLP_hidden_dim', 10)
        nonlinearity = config.get('MLP_nonlinearity', 'ReLU')

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.features.size(-1), n_hidden),
                                        eval(f'torch.nn.{nonlinearity}()'),
                                        torch.nn.Linear(n_hidden, 1)
                                       )

        self.learning_rate = config.get('lr', 0.01)

    def training_step(self, train_mask, *args):

        y_hat = self.mlp(self.features)

        loss = torch.pow(self.targets.view(1, -1)[train_mask] - y_hat.view(1, -1)[train_mask], 2).mean()

        self.log("train_mse", loss.item(), sync_dist=True)

        return loss


    def evaluate(self, test_mask, split='test'):

        y_hat = self.mlp(self.features).squeeze()
        targets = self.targets[test_mask]
        residuals = targets - y_hat[test_mask]

        self.log(f"{split}_mae", residuals.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_mse", torch.pow(residuals, 2).mean().item(), sync_dist=True)
        self.log(f"{split}_rmse", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)
        self.log(f"{split}_mape", (residuals / targets).abs().mean().item(), sync_dist=True)

    def test_step(self, test_mask, *args):

        self.evaluate(test_mask.squeeze(), split='test')


    def validation_step(self, val_mask, *args):

        # y_hat = self.mlp(self.features)
        # val_mask = val_mask.reshape(-1)
        #
        # targets = self.targets[val_mask]
        # residuals = y_hat[val_mask] - targets
        #
        # self.log("val_mae", residuals.abs().mean().item(), sync_dist=True)
        # self.log("val_rmse", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)
        # self.log("val_mse", torch.pow(residuals, 2).mean().item(), sync_dist=True)
        # self.log("val_mape", (residuals / targets).abs().mean().item(), sync_dist=True)

        self.evaluate(val_mask.squeeze(), split='val')


    def predict_step(self, predict_mask, *args):

        y_hat = self.mlp(self.features)

        return {'pred_mean': y_hat,
                'targets': self.targets,
                'predict_mask': predict_mask}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class BayesianLR(pl.LightningModule):

    def __init__(self, config, targets, features, **kwargs):
        super(BayesianLR, self).__init__()

        self.targets = targets # shape [T * num_nodes]
        self.features = features # shape [T * num_nodes, num_features]

        self.model = BayesianRidge()

    def fit(self, train_mask, *args):

        self.model = self.model.fit(self.features[train_mask], self.targets[train_mask])


    def test_step(self, test_mask, *args):

        y_hat, std = self.model.predict(self.features.detach().numpy(), return_std=True)

        mask = test_mask.detach().numpy().reshape(-1)
        targets = self.targets.detach().numpy()
        residuals = y_hat[mask] - targets[mask]

        self.log("test_mae", np.abs(residuals).mean(), sync_dist=True)
        self.log("test_rmse", np.sqrt(np.power(residuals, 2).mean()), sync_dist=True)
        self.log("test_mse", np.power(residuals, 2).mean(), sync_dist=True)
        self.log("test_mape", np.abs(residuals / targets[mask]).mean(), sync_dist=True)

        self.log("test_crps", crps_score(y_hat[mask], std[mask], targets[mask]), sync_dist=True)
        self.log("test_int_score", int_score(y_hat[mask], std[mask], targets[mask]), sync_dist=True)


    def validation_step(self, val_mask, *args):

        y_hat, std = self.model.predict(self.features.detach().numpy(), return_std=True)

        mask = val_mask.detach().numpy().reshape(-1)
        targets = self.targets.detach().numpy()
        residuals = y_hat[mask] - targets[mask]

        self.log("val_mae", np.abs(residuals).mean(), sync_dist=True)
        self.log("val_rmse", np.sqrt(np.power(residuals, 2).mean()), sync_dist=True)
        self.log("val_mse", np.power(residuals, 2).mean(), sync_dist=True)
        self.log("val_mape", np.abs(residuals / targets[mask]).mean(), sync_dist=True)

        self.log("val_crps", crps_score(y_hat[mask], std[mask], targets[mask]), sync_dist=True)
        self.log("val_int_score", int_score(y_hat[mask], std[mask], targets[mask]), sync_dist=True)


    def predict_step(self, predict_mask, *args):

        y_hat, std = self.model.predict(self.features.detach().numpy(), return_std=True)

        return {'pred_mean': y_hat,
                'pred_std': std,
                'targets': self.targets,
                'predict_mask': predict_mask}



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




