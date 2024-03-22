import torch
import pytorch_lightning as pl
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import BayesianRidge
from stdgmrf.utils import crps_score, int_score
import os
import pickle

from stdgmrf.models.KS import KalmanSmoother, EnsembleKalmanSmoother

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

        self.true_post_mean = kwargs.get('true_post_mean', None)
        self.true_post_std = kwargs.get('true_post_std', None)

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

        if self.true_post_mean is not None:
            target = self.true_post_mean.reshape(-1)[mask].detach().numpy()
            residuals = target - x_smoothed
            self.log(f"{split}_mae_mean_smoothed", np.abs(residuals).mean(), sync_dist=True)
            self.log(f"{split}_rmse_mean_smoothed", np.sqrt(np.square(residuals).mean()), sync_dist=True)

            self.log(f"{split}_crps_mean_smoothed", crps_score(x_smoothed, std_smoothed, target), sync_dist=True)
            self.log(f"{split}_int_score_mean_smoothed", int_score(x_smoothed, std_smoothed, target), sync_dist=True)

            if self.true_post_std is not None:
                target_std = self.true_post_std.reshape(-1)[mask].detach().numpy()
                residuals_std = target_std - std_smoothed
                self.log(f"{split}_mae_std_smoothed", np.abs(residuals_std).mean(), sync_dist=True)
                self.log(f"{split}_rmse_std_smoothed", np.sqrt(np.square(residuals_std).mean()), sync_dist=True)


    def test_step(self, test_mask, *args):
        if self.final_prediction:
            # use all data except for test data
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
            data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask.flatten()).to(self.mask.device))
        else:
            # only use dedicated training data (neither validation nor test data)
            data_mask = self.train_mask

        predict_nodes = predict_mask.reshape(self.T, -1).sum(0).nonzero().flatten()
        results = self.forward(data_mask, predict_nodes)

        return results


class KS_EM(pl.LightningModule):

    def __init__(self, config, data, joint_mask, train_mask, T=1, gt=None, **kwargs):
        super(KS_EM, self).__init__()

        self.T = T
        self.y = data
        self.mask = joint_mask
        self.inv_mask = torch.logical_not(self.mask).reshape(T, -1)
        self.num_nodes = self.inv_mask.size(1)
        self.y_masked = torch.ones(self.mask.size(), dtype=self.y.dtype) * np.nan
        self.y_masked[self.mask] = self.y

        self.config = config

        self.train_mask = train_mask
        self.final_prediction = False # set to True for final model fitting and prediction

        if gt is not None:
            self.gt = gt
        else:
            self.gt = self.y_masked

        self.true_post_mean = kwargs.get('true_post_mean', None)
        self.true_post_std = kwargs.get('true_post_std', None)

        initial_mean = torch.zeros(1, self.num_nodes)
        initial_cov = 10 * torch.eye(self.num_nodes).unsqueeze(0)
        #transition_cov = 3 * torch.eye(self.num_nodes).unsqueeze(0)
        transition_cov = torch.diag(3 + torch.rand(self.num_nodes)).unsqueeze(0)
        transition_cov = torch.diag(5 + torch.rand(self.num_nodes)).unsqueeze(0)
        transition_model = 1 * torch.eye(self.num_nodes).unsqueeze(0)
        observation_models = self.all_H(train_mask)
        observation_noise = config.get('noise_std')

        self.KS = KalmanSmoother(initial_mean, initial_cov, transition_model, transition_cov,
         observation_models, observation_noise)


    def all_H(self, mask):
        # mask has shape [T * num_nodes]
        # output is list of T tensors of shape [num_observed_nodes_t, num_nodes]

        identity = torch.eye(self.num_nodes)
        all_H = []

        mask = mask.reshape(self.T, -1)

        # adjust for each time step
        for t in range(self.T):
            # get observed nodes for time t
            jdx = (mask[t].to(torch.float32).flatten()).nonzero().squeeze()

            all_H.append(identity[jdx, :].unsqueeze(0))

        return all_H

    def forward(self, data_mask, n_iterations=0):

        y_masked = torch.zeros_like(self.y_masked)
        y_masked[data_mask] = self.y_masked[data_mask]
        y_masked = y_masked.reshape(self.T, self.num_nodes)

        # update observation model according to mask
        observation_models = self.all_H(data_mask)
        self.KS.update_H(observation_models)

        # estimate parameters with EM algorithm
        self.KS.EM(y_masked.unsqueeze(0), n_iterations, update=['mean', 'alpha', 'Q'], eps=self.config.get('EM_eps', 1e-3))

        # estimate states with Kalman smoother
        mean_smoothed, cov_smoothed, cov_lagged = self.KS.smoother(y_masked.unsqueeze(0))

        std_smoothed = torch.diagonal(cov_smoothed, dim1=2, dim2=3)

        return {'post_mean': mean_smoothed.reshape(self.T, self.num_nodes),
                'post_std': std_smoothed.reshape(self.T, self.num_nodes),
                'post_cov': cov_smoothed.reshape(self.T, self.num_nodes, self.num_nodes),
                'data': y_masked,
                'train_mask': data_mask,
                'gt': self.gt if hasattr(self, 'gt') else 'NA'}

    def evaluate(self, results, mask, split='test'):

        x_smoothed = results['post_mean'].flatten()[mask]
        std_smoothed = results['post_std'].flatten()[mask]

        print(std_smoothed.min(), std_smoothed.max())

        targets = self.gt[mask]
        residuals = targets - x_smoothed

        self.log(f"{split}_mae", residuals.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_mse", torch.pow(residuals, 2).mean().item(), sync_dist=True)
        self.log(f"{split}_rmse", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)
        self.log(f"{split}_mape", (residuals / targets).abs().mean().item(), sync_dist=True)

        x_smoothed_np = x_smoothed.to('cpu').detach().numpy()
        std_smoothed_np = std_smoothed.to('cpu').detach().numpy()
        targets_np = targets.to('cpu').detach().numpy()

        self.log(f"{split}_crps", crps_score(x_smoothed_np, std_smoothed_np, targets_np), sync_dist=True)
        self.log(f"{split}_int_score", int_score(x_smoothed_np, std_smoothed_np, targets_np), sync_dist=True)

        if self.true_post_mean is not None:
            target = self.true_post_mean.reshape(-1)[mask]
            residuals = target - x_smoothed
            self.log(f"{split}_mae_mean", residuals.abs().mean(), sync_dist=True)
            self.log(f"{split}_rmse_mean", torch.pow(residuals, 2).mean().sqrt(), sync_dist=True)

            if self.true_post_std is not None:
                target_std = self.true_post_std.reshape(-1)[mask]
                residuals_std = target_std - std_smoothed
                self.log(f"{split}_mae_std", residuals_std.abs().mean(), sync_dist=True)
                self.log(f"{split}_rmse_std", torch.pow(residuals_std, 2).mean().sqrt(), sync_dist=True)


    def test_step(self, test_mask, *args):
        if self.final_prediction:
            # use all data except for test data
            data_mask = torch.logical_and(self.mask, torch.logical_not(test_mask.flatten()).to(self.mask.device))
            split = 'test'
        else:
            # only use dedicated training data (neither validation nor test data)
            data_mask = self.train_mask
            split = 'val'

        results = self.forward(data_mask, n_iterations=self.config['n_iterations'])
        self.evaluate(results, test_mask.flatten(), split=split)


    def predict_step(self, predict_mask, *args):

        if self.final_prediction:
            # use all data except for test data
            data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask.flatten()).to(self.mask.device))
        else:
            # only use dedicated training data (neither validation nor test data)
            data_mask = self.train_mask

        results = self.forward(data_mask, n_iterations=self.config['n_iterations'])

        return results


class EnKS(pl.LightningModule):

    def __init__(self, config, data, joint_mask, train_mask, transition_model, ensemble_size,
                 initial_params=None, initial_std_params=None, T=1, gt=None, **kwargs):
        super(EnKS, self).__init__()

        self.T = T
        self.y = data
        self.mask = joint_mask
        self.inv_mask = torch.logical_not(self.mask).reshape(T, -1)
        self.num_nodes = self.inv_mask.size(1)
        self.y_masked = torch.zeros(self.mask.size(), dtype=self.y.dtype)
        self.y_masked[self.mask] = self.y

        self.config = config

        self.train_mask = train_mask
        self.final_prediction = False # set to True for final model fitting and prediction

        if gt is not None:
            self.gt = gt
        else:
            self.gt = self.y_masked

        self.true_post_mean = kwargs.get('true_post_mean', None)
        self.true_post_std = kwargs.get('true_post_std', None)

        # TODO: include model parameters for joint parameter and state estimation

        initial_mean = torch.ones(self.num_nodes) * data.mean()

        initial_std = config.get('initial_std_states', 10) * torch.ones(self.num_nodes)
        transition_std = config.get('transition_std_states', 1.7) * torch.ones(self.num_nodes)

        if initial_params is not None:
            self.n_params = len(initial_params)
            initial_mean = torch.cat([initial_mean, initial_params], dim=0)
            self.y_masked = torch.cat([self.y_masked.reshape(T, -1), torch.zeros(T, self.n_params)], dim=1).flatten()
            self.num_nodes = self.num_nodes + self.n_params

            if initial_std_params is None:
                initial_std_params = 0.1 * torch.ones(self.n_params)
            initial_std = torch.cat([initial_std, initial_std_params])
            transition_std = torch.cat([transition_std, config.get('transition_std_params', 1e-5) * torch.ones(self.n_params)])
        else:
            self.n_params = 0

        initial_cov_factor = torch.diag(initial_std)
        transition_cov_factor = torch.diag(transition_std)


        observation_models = self.all_H(train_mask)
        observation_noise = config.get('noise_std')

        self.EnKS = EnsembleKalmanSmoother(ensemble_size, initial_mean, initial_cov_factor, transition_model,
                                           transition_cov_factor, observation_models, observation_noise)


    def all_H(self, mask):
        # mask has shape [T * num_nodes]
        # output is list of T tensors of shape [num_observed_nodes_t, num_nodes]

        identity = torch.eye(self.num_nodes)
        all_H = []

        mask = mask.reshape(self.T, -1)

        # adjust for each time step
        for t in range(self.T):
            # get observed nodes for time t
            jdx = (mask[t].to(torch.float32).flatten()).nonzero().squeeze()

            all_H.append(identity[jdx, :])

        return all_H

    def forward(self, data_mask):

        if self.n_params > 0:
            data_mask = torch.cat([data_mask.reshape(self.T, -1),
                                   torch.zeros(self.T, self.n_params, dtype=data_mask.dtype)], dim=1).flatten()

        y_masked = torch.zeros_like(self.y_masked)
        y_masked[data_mask] = self.y_masked[data_mask]
        y_masked = y_masked.reshape(self.T, self.num_nodes)

        # update observation model according to mask
        observation_models = self.all_H(data_mask)
        self.EnKS.update_H(observation_models)

        # estimate states with Kalman smoother
        _, _, ensemble_smoothed = self.EnKS.smoother(y_masked)

        cov_smoothed = self.EnKS.ensemble_cov(ensemble_smoothed)

        std_smoothed = torch.diagonal(cov_smoothed, dim1=-2, dim2=-1)
        mean_smoothed = ensemble_smoothed.mean(-1)

        return {'post_mean': mean_smoothed[:, :self.num_nodes - self.n_params],
                'post_std': std_smoothed[:, :self.num_nodes - self.n_params],
                'post_cov': cov_smoothed[:, :self.num_nodes - self.n_params, :self.num_nodes - self.n_params],
                'data': y_masked[:, :self.num_nodes - self.n_params],
                'train_mask': data_mask[:self.num_nodes - self.n_params],
                'gt': self.gt if hasattr(self, 'gt') else 'NA'}

    def evaluate(self, results, mask, split='test'):

        idx = mask.reshape(20, -1).sum(0).nonzero().flatten()[0]
        jdx = torch.logical_not(mask).reshape(20, -1).sum(0).nonzero().flatten()[0]

        print(results['post_mean'].reshape(20, -1)[:, idx])
        print(self.gt.reshape(20, -1)[:, idx])

        print(results['post_mean'].reshape(20, -1)[:, jdx])
        print(self.gt.reshape(20, -1)[:, jdx])

        x_smoothed = results['post_mean'].flatten()[mask]
        std_smoothed = results['post_std'].flatten()[mask]

        targets = self.gt[mask]
        residuals = targets - x_smoothed

        self.log(f"{split}_mae", residuals.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_mse", torch.pow(residuals, 2).mean().item(), sync_dist=True)
        self.log(f"{split}_rmse", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)
        self.log(f"{split}_mape", (residuals / targets).abs().mean().item(), sync_dist=True)

        x_smoothed_np = x_smoothed.to('cpu').detach().numpy()
        std_smoothed_np = std_smoothed.to('cpu').detach().numpy()
        targets_np = targets.to('cpu').detach().numpy()

        self.log(f"{split}_crps", crps_score(x_smoothed_np, std_smoothed_np, targets_np), sync_dist=True)
        self.log(f"{split}_int_score", int_score(x_smoothed_np, std_smoothed_np, targets_np), sync_dist=True)

        if self.true_post_mean is not None:
            target = self.true_post_mean.reshape(-1)[mask]
            residuals = target - x_smoothed
            self.log(f"{split}_mae_mean", residuals.abs().mean(), sync_dist=True)
            self.log(f"{split}_rmse_mean", torch.pow(residuals, 2).mean().sqrt(), sync_dist=True)

            if self.true_post_std is not None:
                target_std = self.true_post_std.reshape(-1)[mask]
                residuals_std = target_std - std_smoothed
                self.log(f"{split}_mae_std", residuals_std.abs().mean(), sync_dist=True)
                self.log(f"{split}_rmse_std", torch.pow(residuals_std, 2).mean().sqrt(), sync_dist=True)


    def test_step(self, test_mask, *args):
        if self.final_prediction:
            # use all data except for test data
            data_mask = torch.logical_and(self.mask, torch.logical_not(test_mask.flatten()).to(self.mask.device))
            split = 'test'
        else:
            # only use dedicated training data (neither validation nor test data)
            data_mask = self.train_mask
            split = 'val'

        results = self.forward(data_mask)
        self.evaluate(results, test_mask.flatten(), split=split)


    def predict_step(self, predict_mask, *args):

        if self.final_prediction:
            # use all data except for test data
            data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask.flatten()).to(self.mask.device))
        else:
            # only use dedicated training data (neither validation nor test data)
            data_mask = self.train_mask

        results = self.forward(data_mask)

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


