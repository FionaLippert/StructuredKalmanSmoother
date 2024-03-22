import torch
from torch.nn.parameter import Parameter
import torch_geometric as ptg
import copy
import numpy as np
import scipy.stats as sps
from stdgmrf import cg_batch
import pytorch_lightning as pl
from timeit import default_timer as timer

from stdgmrf import utils
from stdgmrf.models.transition_models import *
from stdgmrf.models.KS import KalmanSmoother



def get_num_nodes(edge_index):
    return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0


class SpatiotemporalInference(pl.LightningModule):

    def __init__(self, config, initial_guess, data, joint_mask, spatial_graph, temporal_graph=None,
                 T=1, gt=None, features=None, **kwargs):

        super(SpatiotemporalInference, self).__init__()
        self.save_hyperparameters()

        self.config = config

        # model settings
        self.learning_rate = config.get('lr', 0.01)

        self.T = T
        self.num_nodes = get_num_nodes(spatial_graph['edge_index'])
        self.N = self.T * self.num_nodes
        print(f'N={self.N}, T={self.T}')

        self.y = data
        self.mask = joint_mask  # shape [T * num_nodes]
        self.y_masked = torch.zeros(self.mask.size(), dtype=self.y.dtype)  # * np.nan
        self.y_masked[self.mask] = self.y

        self.pos = spatial_graph.get('pos', None)
        if not gt is None:
            self.gt = gt  # shape [T * num_nodes]

        if config["learn_noise_std"]:
            self.obs_noise_param = torch.nn.parameter.Parameter(torch.tensor(config["noise_std"]))
        else:
            self.obs_noise_param = torch.tensor(config["noise_std"])

        self.use_dynamics = config.get("use_dynamics", False)
        self.independent_time = config.get("independent_time", False)
        self.use_hierarchy = config.get("use_hierarchy", False)
        self.use_vi_dynamics = config.get('use_vi_dynamics', False)

        self.features = features

        self.use_features = config.get("use_features", False) and features is not None
        self.use_features_dynamics = config.get("use_features_dynamics", False) and features is not None

        self.data_mean = kwargs.get('data_mean', 0)
        self.data_std = kwargs.get('data_std', 1)

        self.true_post_mean = kwargs.get('true_post_mean', None)
        self.true_post_std = kwargs.get('true_post_std', None)

        # model components
        if self.use_dynamics:

            shared = 'none' if config.get('independent_time', True) else 'dynamics'
            features = self.features.reshape(self.T, self.num_nodes, -1) if self.use_features_dynamics else None

            self.dgmrf = JointDGMRF(config, spatial_graph, temporal_graph, T=self.T, shared=shared,
                                    weighted=config.get('weighted_dgmrf', False), features=features)

            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'none'

        elif self.independent_time:
            # treat time steps independently, with separate DGMRF for each time step
            self.dgmrf = DGMRF(config, spatial_graph, T=self.T, shared='none',
                               weighted=config.get('weighted_dgmrf', False))
            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'none'
        else:
            # use a single DGMRF, with parameters shared across time steps
            self.dgmrf = DGMRF(config, spatial_graph, T=self.T, shared='all',
                               weighted=config.get('weighted_dgmrf', False))
            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'all'

        if not config.get('use_KS', False):
            self.vi_dist = VariationalDist(config, spatial_graph, initial_guess.reshape(self.T, -1),
                                           T=self.T, shared=shared_vi,
                                           temporal_graph=(temporal_graph if self.use_vi_dynamics else None),
                                           n_features=(self.features.size(-1) if self.use_features else 0))

        self.obs_model = lambda x: x[:, self.mask]

        self.n_training_samples = config["n_training_samples"]

    def get_name(self):
        return 'SpatiotemporalDGMRF'

    @property
    def noise_var(self):
        return self.obs_noise_param ** 2

    @property
    def log_noise_std(self):
        return 0.5 * self.noise_var.log()

    def all_H(self, mask):
        # mask has shape [T * num_nodes]

        identity = torch.eye(self.num_nodes)
        all_H = []

        # adjust for each time step
        for t in range(self.T):
            # get observed nodes for time t
            jdx = mask.view(self.T, -1)[t].to(torch.float32).flatten().nonzero().squeeze()

            all_H.append(identity[jdx, :].unsqueeze(0))

        return all_H

    def stacked_H(self, mask):
        # mask has shape [T * num_nodes]
        # output should have shape [T, num_observed_nodes, num_nodes]
        obs_nodes = mask.reshape(self.T, -1).sum(0).nonzero().squeeze()

        stacked_H = torch.eye(self.num_nodes).unsqueeze(0).repeat(self.T, 1, 1)
        stacked_H = stacked_H[:, obs_nodes, :]  # nodes that are observed at any time point

        sub_mask = (stacked_H @ mask.view(self.T, -1, 1).to(torch.float32)).squeeze(-1)
        unobs_nodes = (sub_mask - 1).nonzero()

        # adjust for each time step
        for tidx, nidx in unobs_nodes:
            # get unobserved nodes for time t
            stacked_H[tidx, nidx] = 0

        return stacked_H

    def _reconstruction_loss(self, x, mask):
        # x has shape [n_samples, T * n_nodes]

        rec_loss = torch.sum(torch.pow(self.y_masked[..., mask] - x[..., mask], 2))

        return rec_loss

    def _joint_log_likelihood(self, x, mask, v=None):
        # x has shape [n_samples, T, n_nodes]
        # v has shape [2, n_samples, n_nodes]
        # N = x.size(-1) * x.size(-2)

        # compute log-likelihood of samples given prior p(x)
        Gx = self.dgmrf(x, v=v)  # shape (n_samples, T, n_nodes)

        prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) / self.n_training_samples + self.dgmrf.log_det())

        x = x.reshape(self.n_training_samples, -1)  # shape [n_samples, T * num_nodes]

        # compute data log-likelihood given samples
        if self.use_features:
            coeffs = self.vi_dist.sample_coeff(self.n_training_samples).unsqueeze(1)  # [n_samples, 1, n_features]
            x = x + coeffs @ self.features.transpose(0, 1)

        rec_loss = self._reconstruction_loss(x, mask) / self.n_training_samples

        data_ll = -0.5 * rec_loss / self.noise_var - self.log_noise_std * mask.sum()

        self.log("train_rec_loss", rec_loss.item(), sync_dist=True)
        self.log("train_prior_ll", prior_ll.item(), sync_dist=True)
        self.log("train_data_ll", data_ll.item(), sync_dist=True)

        return prior_ll + data_ll

    def KS_ELBO(self, mu_0, Q_0, mu_s, cov_s, cov_s_lag, F_t, Q_t, data, stacked_H):
        # mu_0 has shape [num_nodes]
        # Q_0 has shape [num_nodes, num_nodes]
        # mu_s has shape [T, num_nodes]
        # cov_s has shape [T, num_nodes, num_nodes]
        # cov_s_lag has shape[T-1, num_nodes, num_nodes]
        # F_t, Q_t have shape [T-1, num_nodes, num_nodes]

        # compute E_{x|y}[p(x)]
        mu_0 = mu_0.unsqueeze(-1)
        mu_s = mu_s.unsqueeze(-1)
        E0 = torch.trace(Q_0 @ (cov_s[0] + mu_s[0] @ mu_s[0].transpose(0, 1))) \
             - 2 * mu_0.transpose(0, 1) @ Q_0 @ mu_s[0] \
             + mu_0.transpose(0, 1) @ Q_0 @ mu_0

        Et = (Q_t @ (cov_s[1:] + mu_s[1:] @ mu_s[1:].transpose(-1, -2))).diagonal(offset=0, dim1=-1, dim2=-2) \
             - 2 * (Q_t @ F_t @ (cov_s_lag + mu_s[:-1] @ mu_s[1:].transpose(-1, -2))).diagonal(offset=0, dim1=-1,
                                                                                               dim2=-2) \
             + (Q_t @ F_t @ (cov_s[:-1] + mu_s[:-1] @ mu_s[:-1].transpose(-1, -2))).diagonal(offset=0, dim1=-1, dim2=-2)

        prior_term = -0.5 * (E0 + Et.sum()) + self.dgmrf.log_det()

        # compute E_{x|y}[p(y|x)]
        data = data.unsqueeze(-1)
        data_term = -0.5 * (((stacked_H @ data).transpose(-1, -2) @ stacked_H @ data) / self.noise_var \
                            + ((stacked_H @ (cov_s + mu_s @ mu_s.transpose(-1, -2)) @ stacked_H.transpose(-1, -2)) \
                               / self.noise_var).diagonal(offset=0, dim1=-1, dim2=-2)
                            - (2 * (stacked_H @ data).transpose(-1, -2) @ stacked_H @ mu_s) / self.noise_var).sum() \
                    - stacked_H.sum() * self.log_noise_std

        return prior_term + data_term

    def training_step(self, train_mask):

        self.train_mask = train_mask.squeeze(0)

        if self.config.get('use_KS', False):
            Q_0, Q_t = self.dgmrf.get_matrices()  # all have shape [num_nodes, num_nodes]
            # TODO: allow for time-varying F_t (depending on covariates)?
            F_t = self.dgmrf.get_transition_matrix()
            # assume mu_0 and c_t are all zero
            # TODO: get mu_0 and c_t from STDGMRF
            mu_0 = torch.zeros(self.num_nodes)
            # R_t = self.noise_var * torch.eye(self.num_nodes)

            Q_0_inv, Q_t_inv = self.dgmrf.get_inv_matrices()

            data = (self.y_masked * self.train_mask).reshape(self.T, -1)

            all_H = self.all_H(self.train_mask)
            stacked_H = self.stacked_H(self.train_mask)

            ks = KalmanSmoother(mu_0.unsqueeze(0), Q_0_inv.unsqueeze(0), F_t.unsqueeze(0),
                                Q_t_inv.unsqueeze(0), all_H, torch.ones(1, ) * self.noise_var)
            mu_s, cov_s, cov_s_lag = ks.smoother(data.unsqueeze(0))

            Q_t = Q_t.unsqueeze(0).repeat(self.T - 1, 1, 1)
            F_t = F_t.unsqueeze(0).repeat(self.T - 1, 1, 1)

            elbo = self.KS_ELBO(mu_0, Q_0, mu_s.squeeze(0), cov_s.squeeze(0), cov_s_lag.squeeze(0),
                                F_t, Q_t, data, stacked_H)

            self.log("train_elbo", elbo.item(), sync_dist=True)

        else:
            # sample from variational distribution
            samples = self.vi_dist.sample()  # shape [n_samples, T, num_nodes]

            # compute entropy of variational distribution
            vi_entropy = 0.5 * self.vi_dist.log_det()

            joint_ll = self._joint_log_likelihood(samples, self.train_mask)

            if self.use_features:
                vi_entropy = vi_entropy + 0.5 * self.vi_dist.log_det_coeff()  # log-det for coefficients
                joint_ll = joint_ll + self.vi_dist.ce_coeff()  # cross entropy term for coefficients

            elbo = (joint_ll + vi_entropy) / self.N

            self.log("train_elbo", elbo.item(), sync_dist=True)
            self.log("vi_entropy", vi_entropy.item(), sync_dist=True)

        torch.cuda.empty_cache()

        return -1. * elbo

    def validation_step(self, val_mask, *args):

        torch.cuda.empty_cache()

        if self.config.get('use_KS', False):
            pass
        else:
            samples = self.vi_dist.sample()  # shape [n_samples, T, num_nodes]
            samples = samples.reshape(self.n_training_samples, -1)  # shape [n_samples, T * num_nodes]

            if self.use_features:
                coeffs = self.vi_dist.sample_coeff(self.n_training_samples).unsqueeze(1)  # [n_samples, 1, n_features]
                samples = samples + coeffs @ self.features.transpose(0, 1)

            rec_loss = self._reconstruction_loss(samples, val_mask.squeeze(0)) / (
                        val_mask.sum() * self.n_training_samples)

            self.log("val_rec_loss", rec_loss.item(), sync_dist=True)

    def test_step(self, test_mask, *args):

        torch.cuda.empty_cache()

        if self.config.get('use_KS', False):
            self.KS_inference(test_mask.squeeze(0), split='test')
        else:
            self.DGMRF_inference(test_mask.squeeze(0), split='test')

    def DGMRF_inference(self, test_mask, split='test'):

        # posterior inference using variational distribution
        self.vi_mean, self.vi_std = self.vi_dist.posterior_estimate(self.noise_var)
        mean = self.vi_mean.flatten()
        std = self.vi_std.flatten()

        # use held out part of data to evaluate predictions
        data_mask = torch.logical_and(self.mask, torch.logical_not(test_mask))  # all data except test data

        y_masked = self.y_masked * data_mask

        self.post_mean, self.post_std, niter = posterior_inference(self.dgmrf, y_masked.reshape(1, *self.input_shape),
                                                                   data_mask, self.config, self.noise_var, self.vi_mean,
                                                                   features=self.features, verbose=False,
                                                                   preconditioner=None)

        self.log('niter_cg', niter, sync_dist=True)

        if hasattr(self, 'gt'):
            target = self.gt[test_mask]
        else:
            target = self.y_masked[test_mask]

        residuals_vi = (target - mean[test_mask])
        residuals_cg = target - self.post_mean.squeeze(-1)[test_mask]

        self.log(f"{split}_mae_vi", residuals_vi.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_rmse_vi", torch.pow(residuals_vi, 2).mean().sqrt().item(), sync_dist=True)

        self.log(f"{split}_mae", residuals_cg.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_rmse", torch.pow(residuals_cg, 2).mean().sqrt().item(), sync_dist=True)

        pred_mean_np = self.post_mean.squeeze(-1)[test_mask].cpu().numpy()
        pred_std_np = self.post_std.squeeze(-1)[test_mask].cpu().numpy()
        target_np = target.cpu().numpy()

        self.log(f"{split}_crps", utils.crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
        self.log(f"{split}_int_score", utils.int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)

        vi_mean_np = self.vi_mean.flatten()[test_mask].cpu().numpy()
        vi_std_np = self.vi_std.flatten()[test_mask].cpu().numpy()

        self.log(f"{split}_crps_vi", utils.crps_score(vi_mean_np, vi_std_np, target_np), sync_dist=True)
        self.log(f"{split}_int_score_vi", utils.int_score(vi_mean_np, vi_std_np, target_np), sync_dist=True)

        vi_mean_np = self.vi_mean.flatten()[test_mask].cpu().numpy()
        vi_std_np = self.vi_std.flatten()[test_mask].cpu().numpy()
        self.log(f"{split}_crps_vi", utils.crps_score(vi_mean_np, vi_std_np, target_np), sync_dist=True)

        if self.true_post_mean is not None:

            target = self.true_post_mean.reshape(-1)[test_mask]
            residuals = target - self.post_mean.squeeze(-1)[test_mask]
            self.log(f"{split}_mae_mean", residuals.abs().mean().item(), sync_dist=True)
            self.log(f"{split}_rmse_mean", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)

            residuals_vi = (target - mean[test_mask])
            self.log(f"{split}_mae_mean_vi", residuals_vi.abs().mean().item(), sync_dist=True)
            self.log(f"{split}_rmse_mean_vi", torch.pow(residuals_vi, 2).mean().sqrt().item(), sync_dist=True)

            target_np = target.cpu().numpy()

            self.log(f"{split}_crps_mean", utils.crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
            self.log(f"{split}_int_score_mean", utils.int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)

            self.log(f"{split}_crps_mean_vi", utils.crps_score(vi_mean_np, vi_std_np, target_np), sync_dist=True)
            self.log(f"{split}_int_score_mean_vi", utils.int_score(vi_mean_np, vi_std_np, target_np), sync_dist=True)

            if self.true_post_std is not None:
                target_std = self.true_post_std.reshape(-1)[test_mask]
                residuals_std = target_std - self.post_std.squeeze(-1)[test_mask]
                self.log(f"{split}_mae_std", residuals_std.abs().mean().item(), sync_dist=True)
                self.log(f"{split}_rmse_std", torch.pow(residuals_std, 2).mean().sqrt().item(), sync_dist=True)

                residuals_std_vi = target_std - self.vi_std.flatten()[test_mask]
                self.log(f"{split}_mae_std_vi", residuals_std_vi.abs().mean().item(), sync_dist=True)
                self.log(f"{split}_rmse_std_vi", torch.pow(residuals_std_vi, 2).mean().sqrt().item(), sync_dist=True)

    def predict_step(self, predict_mask, *args):

        if self.config.get('use_KS', False):
            return self.KS_prediction(predict_mask.squeeze(0))
        else:
            return self.DGMRF_prediction(predict_mask.squeeze(0))

    def DGMRF_prediction(self, predict_mask):

        data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask))

        y_masked = torch.zeros_like(self.y_masked)
        y_masked[data_mask] = self.y_masked[data_mask]

        if not (hasattr(self, 'vi_mean') or hasattr(self, 'vi_std')):
            self.vi_mean, self.vi_std = self.vi_dist.posterior_estimate(self.noise_var)

        if not (hasattr(self, 'cg_mean') or hasattr(self, 'cg_std')):
            self.post_mean, self.post_std, niter = posterior_inference(self.dgmrf,
                                                                       y_masked.reshape(1, *self.input_shape),
                                                                       data_mask, self.config, self.noise_var,
                                                                       self.vi_mean,
                                                                       features=self.features)

        if True:  # not hasattr(self, 'post_samples'):
            print('sample from posterior')
            initial_guess = self.vi_mean
            self.post_samples = sample_posterior(10, self.dgmrf, y_masked.reshape(1, *self.input_shape),
                                                 data_mask, self.config, self.noise_var, initial_guess,
                                                 features=self.features)

        results = {'post_mean': self.post_mean.reshape(self.T, self.num_nodes),
                   'post_std': self.post_std.reshape(self.T, self.num_nodes),
                   'post_samples': self.post_samples.reshape(-1, self.T, self.num_nodes),
                   'vi_mean': self.vi_mean,
                   'vi_std': self.vi_std,
                   'data': y_masked.reshape(self.T, self.num_nodes),
                   'gt': self.gt if hasattr(self, 'gt') else self.y_masked,
                   'predict_mask': predict_mask}

        if self.config.get('save_transition_matrix', False):
            print('save transition matrix')
            results['transition_matrix'] = self.dgmrf.get_transition_matrix()

        return results

    def KS_inference(self, test_mask, split):
        # use held out part of data to evaluate predictions
        data_mask = torch.logical_and(self.mask, torch.logical_not(test_mask))  # all data except test data

        data = (self.y_masked * data_mask).reshape(self.T, -1)

        # assume mu_0 and c_t are all zero
        mu_0 = torch.zeros(self.num_nodes)

        F_t = self.dgmrf.get_transition_matrix()
        Q_0_inv, Q_t_inv = self.dgmrf.get_inv_matrices()

        all_H = self.all_H(test_mask)

        ks = KalmanSmoother(mu_0.unsqueeze(0), Q_0_inv.unsqueeze(0), F_t.unsqueeze(0),
                            Q_t_inv.unsqueeze(0), all_H, torch.ones(1, ) * self.noise_var)
        mu_s, cov_s, cov_s_lag = ks.smoother(data.unsqueeze(0))

        self.post_mean = mu_s.reshape(-1)
        self.post_std = torch.diagonal(cov_s, dim1=-2, dim2=-1).squeeze(0).reshape(-1)

        if hasattr(self, 'gt'):
            target = self.gt[test_mask]
        else:
            target = self.y_masked[test_mask]

        residuals = (target - self.post_mean[test_mask])

        self.log(f"{split}_mae", residuals.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_rmse", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)
        self.log(f"{split}_mape", (residuals / target).abs().mean().item(), sync_dist=True)

        pred_mean_np = self.post_mean[test_mask].cpu().numpy()
        pred_std_np = self.post_std[test_mask].cpu().numpy()
        target_np = target.cpu().numpy()

        self.log(f"{split}_crps", utils.crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
        self.log(f"{split}_int_score", utils.int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)

    def KS_prediction(self, predict_mask):
        # assume mu_0 and c_t are all zero

        data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask))

        data = torch.zeros_like(self.y_masked)
        data[data_mask] = self.y_masked[data_mask]
        data = data.reshape(self.T, -1)

        mu_0 = torch.zeros(self.num_nodes)

        F_t = self.dgmrf.get_transition_matrix()
        Q_0_inv, Q_t_inv = self.dgmrf.get_inv_matrices()

        all_H = self.all_H(predict_mask)

        ks = KalmanSmoother(mu_0.unsqueeze(0), Q_0_inv.unsqueeze(0), F_t.unsqueeze(0),
                            Q_t_inv.unsqueeze(0), all_H, torch.ones(1, ) * self.noise_var)
        mu_s, cov_s, cov_s_lag = ks.smoother(data.unsqueeze(0))

        return {'ks_mean': mu_s.squeeze(0),
                'ks_std': torch.diagonal(cov_s, dim1=-2, dim2=-1).squeeze(0),
                'data': data,
                'gt': self.gt if hasattr(self, 'gt') else self.y_masked,
                'predict_mask': predict_mask}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer



class VariationalDist(torch.nn.Module):
    def __init__(self, config, graph, initial_guess, temporal_graph=None, n_features=0, T=1, shared='all'):
        super().__init__()

        # Dimensionality of distribution (num_nodes of graph)
        self.dim = get_num_nodes(graph['edge_index'])
        self.T = T

        # Standard amount of samples (must be fixed to be efficient)
        self.n_samples = config["n_training_samples"]
        self.n_post_samples = config["n_post_samples"]

        # Variational distribution, Initialize with observed y
        self.mean_param = torch.nn.parameter.Parameter(initial_guess)
        self.diag_param = torch.nn.parameter.Parameter(2 * torch.rand(self.T, self.dim) - 1.)  # U(-1,1)
        self.layers = torch.nn.ModuleList([DGMRFLayerMultiChannel(config, graph, T=T, shared=shared, vi_layer=True,
                                                                  weighted=config.get('weighted_vi', False))
                            for _ in range(config["vi_layers"])])

        if temporal_graph is not None:
            self.dynamics = TemporalDGMRF(config, graph, temporal_graph, T=T, shared=shared, vi_layer=True)


        if config["vi_layers"] > 0:
            self.post_diag_param = torch.nn.parameter.Parameter(
                2 * torch.rand(self.T, self.dim) - 1.)

        if config["use_features"] and n_features > 0:
            # Additional variational distribution for linear coefficients
            self.coeff_mean_param = torch.nn.parameter.Parameter(torch.randn(1, n_features))
            self.coeff_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(1, n_features) - 1.) # U(-1,1)

            self.coeff_inv_std = config["coeff_inv_std"]

    @property
    def std(self):
        # Note: Only std before layers
        return torch.nn.functional.softplus(self.diag_param)

    @property
    def post_diag(self):
        # Diagonal of diagonal matrix applied after layers
        return torch.nn.functional.softplus(self.post_diag_param)

    @property
    def coeff_std(self):
        return torch.nn.functional.softplus(self.coeff_diag_param)

    def sample(self):
        # standard_sample = torch.randn(self.n_samples, self.dim)
        samples = torch.randn(self.n_samples, self.T, self.dim)

        if hasattr(self, 'dynamics'):
            samples = samples - self.dynamics(samples, with_bias=False)

        samples = self.std * samples # [T, dim] * [samples, T, dim]

        for layer in self.layers:
            propagated = layer(samples, transpose=False, with_bias=False)
            samples = propagated

        if self.layers:
            # Apply post diagonal matrix
            samples = self.post_diag.unsqueeze(0) * samples # [1, T, dim] * [samples, T, dim]
        samples = samples + self.mean_param.unsqueeze(0) # [samples, T, dim] + [1, T, dim]
        return samples # shape (n_samples, T, dim)

    def forward(self, x, transpose=False):

        if transpose:
            return self.P_transpose(x)
        else:
            return self.P(x)

    def P(self, x):
        # apply factor P to vector x
        # x has shape [nbatch, T, dim]

        if hasattr(self, 'dynamics'):
            # multiply with \Tilde{F}
            out = x - self.dynamics(x, with_bias=False)
        else:
            out = x

        # multiply with diagonal matrix
        out = self.std * out  # [T, dim] * [nbatch, T, dim]

        # multiply with \Tilde{S}
        for layer in self.layers:
            out = layer(out, transpose=False, with_bias=False)

        if self.layers:
            # multiply with post diagonal matrix
            out = self.post_diag.unsqueeze(0) * out # [1, T, dim] * [nbatch, T, dim]

        return out # shape (nbatch, T, dim)


    def P_transpose(self, x):
        # apply factor P^T to vector x
        # x has shape [nbatch, T, dim]

        if self.layers:
            # multiply with post diagonal matrix
            out = self.post_diag.unsqueeze(0) * x # [1, T, dim] * [nbatch, T, dim]
        else:
            out = x

        # multiply with \Tilde{S}
        for layer in self.layers:
            out = layer(out, transpose=True, with_bias=False)

        # multiply with diagonal matrix
        out = self.std * out # [T, dim] * [nbatch, T, dim]

        if hasattr(self, 'dynamics'):
            # multiply with \Tilde{F}
            out = out - self.dynamics(out, transpose=True, with_bias=False)

        return out # shape (nbatch, T, dim)


    def log_det(self):
        layers_log_det = sum([layer.log_det() for layer in self.layers])
        std_log_det = torch.sum(torch.log(self.std))
        total_log_det = 2.0*std_log_det + 2.0*layers_log_det

        if self.layers:
            post_diag_log_det = torch.sum(torch.log(self.post_diag))
            total_log_det = total_log_det + 2.0*post_diag_log_det

        return total_log_det

    def sample_coeff(self, n_samples):
        standard_sample = torch.randn(n_samples, *self.coeff_mean_param.shape)
        samples = (self.coeff_std * standard_sample) + self.coeff_mean_param
        return samples # [n_samples, 1, n_features]

    def log_det_coeff(self):
        return 2.0*torch.sum(torch.log(self.coeff_std))

    def ce_coeff(self):
        # Compute Cross-entropy term (CE between VI-dist and coeff prior)
        return -0.5*(self.coeff_inv_std**2)*torch.sum(
                torch.pow(self.coeff_std, 2) + torch.pow(self.coeff_mean_param, 2))

    @torch.no_grad()
    def posterior_estimate(self, noise_var):
        # Compute mean and marginal std of distribution (posterior estimate)
        # only for x-part, not for beta-part (linear model for covariates)

        # Marginal std. (MC estimate)
        mc_sample_list = []
        cur_mc_samples = 0
        while cur_mc_samples < self.n_post_samples:
            mc_sample_list.append(self.sample())
            cur_mc_samples += self.n_samples
        mc_samples = torch.cat(mc_sample_list, dim=0)[:self.n_post_samples]

        # MC estimate of variance using known population mean
        post_var_x = torch.mean(torch.pow(mc_samples - self.mean_param, 2), dim=0)
        # Posterior std.-dev. for y
        post_std = torch.sqrt(post_var_x + noise_var)

        return self.mean_param.detach(), post_std # [channels, dim]

class ObservationModel(ptg.nn.MessagePassing):
    """
    Apply observation model to latent states: y = Hx
    """

    def __init__(self, graph):
        super(ObservationModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.edge_index = graph['edge_index']

        if 'edge_weight' in graph:
            self.edge_weights = graph['edge_weight']
        else:
            self.edge_weights = torch.ones_like(self.edge_index[0])


        self.mask = ptg.utils.degree(self.edge_index[0], num_nodes=get_num_nodes(self.edge_index)).bool()


    def forward(self, x):

        y = self.propagate(self.edge_index, x=x, edge_weights=self.edge_weights)

        return y[:, self.mask]

    def message(self, x_j, edge_weights):
        # construct messages to node i for each edge (j,i)
        msg = edge_weights * x_j
        return msg




class JointDGMRF(torch.nn.Module):
    def __init__(self, config, spatial_graph, temporal_graph=None, T=1, shared='dynamics',
                 weighted=False, features=None):
        super().__init__()

        if config.get('diag_noise_model', False):
            self.dgmrf = DiagonalModel(config, spatial_graph, T=T, shared=shared)
        else:
            self.dgmrf = DGMRF(config, spatial_graph, T=T, shared=shared, weighted=weighted)
        self.T = T

        if temporal_graph is not None and config.get('n_transitions', 0) > 0:
            self.dynamics = torch.nn.ModuleList([
                TemporalDGMRF(config, spatial_graph, temporal_graph, T=T, shared=shared, features=features)
                for _ in range(config.get('n_layers_temporal', 1))])

    def forward(self, x, transpose=False, with_bias=True, **kwargs):
        # x has shape [num_samples, T, num_nodes]

        if not transpose:
            x = self.apply_temporal(x, transpose=False, with_bias=with_bias)

        x = self.apply_spatial(x, transpose=transpose, with_bias=with_bias)

        if transpose:
            x = self.apply_temporal(x, transpose=True, with_bias=with_bias)

        return x

    def apply_temporal(self, x, transpose=False, with_bias=True):
        out = x
        if hasattr(self, 'dynamics'):
            # apply temporal layers
            for p, layer in enumerate(self.dynamics):
                out = out - layer(x, transpose=transpose, with_bias=with_bias, p=p+1)

        return out

    def apply_spatial(self, x, transpose=False, with_bias=True):
        # apply spatial layers
        out = self.dgmrf(x, transpose=transpose, with_bias=with_bias)

        return out

    def get_matrices(self):

        return self.dgmrf.get_matrices()

    def get_inv_matrices(self):

        return self.dgmrf.get_inv_matrices()

    def get_transition_matrix(self, p=1, t_start=0, t_end=-1):
        assert len(self.dynamics) >= p

        return self.dynamics[p-1].get_matrix(t_start=t_start, t_end=t_end)

    def log_det(self):
        return self.dgmrf.log_det()




class TemporalDGMRF(torch.nn.Module):
    def __init__(self, config, spatial_graph, temporal_graph, features=None, **kwargs):
        super().__init__()

        self.num_nodes = get_num_nodes(spatial_graph['edge_index'])

        self.features = features
        self.use_features = config.get('use_features_dynamics', False) and features is not None
        self.n_features = self.features.size(-1) if self.use_features else 0

        self.vi_layer = kwargs.get('vi_layer', False)
        if self.vi_layer:
            self.transition_type = config.get('vi_transition_type', 'identity')
            self.n_transitions = config.get('vi_transitions', 1)
        else:
            self.transition_type = config.get('transition_type', 'identity')
            self.n_transitions = config.get('n_transitions', 1)

        self.shared_dynamics = kwargs.get('shared', 'dynamics')

        # setup transition model
        if self.transition_type == 'diffusion':
            self.transition_models = torch.nn.ModuleList([DiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'directed_diffusion':
            self.transition_models = torch.nn.ModuleList([DirectedDiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'advection':
            self.transition_models = torch.nn.ModuleList([AdvectionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'GNN_advection':
            self.transition_models = torch.nn.ModuleList([GNNAdvection(config, temporal_graph, self.n_features)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'advection+diffusion':
            self.transition_models = torch.nn.ModuleList([AdvectionDiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'inhomogeneous_advection+diffusion':
            self.transition_models = torch.nn.ModuleList([InhomogeneousAdvectionDiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'GNN':
            self.transition_models = torch.nn.ModuleList([GNNTransition(config, temporal_graph,
                                                                        self.n_features, **kwargs)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'flow':
            self.transition_models = torch.nn.ModuleList([FlowModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        else:
            T = 1 if self.shared_dynamics == 'dynamics' else (kwargs.get('T', 2) - 1)
            self.transition_models = torch.nn.ModuleList([ARModelMultiChannel(T=T) for _ in range(self.n_transitions)])

        if kwargs.get('use_dynamics_bias', True) and not self.vi_layer:
            print('use dynamics bias')
            if self.use_features:
                hidden_dim = config.get('MLP_hidden_dim', 8)
                self.bias_net = torch.nn.Sequential(torch.nn.Linear(self.n_features, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, 1))
            else:
                self.bias_param = torch.nn.parameter.Parameter(torch.rand(1,) - 0.5)
        else:
            self.bias_param = torch.zeros(1)

    def bias(self, features=None):
        if self.use_features:
            return self.bias_net(features)
        else:
            return self.bias_param

    def forward(self, x, features=None, transpose=False, with_bias=True, p=1, **kwargs):
        # computes e=Fx
        # x has shape [n_samples, T, num_nodes]
        # p is the auto-regressive order

        if features is None:
            features = self.features

        if transpose:
            states = x[:, p:]
            features = features[p:] if self.use_features else None
            layers = reversed(self.transition_models)
        else:
            states = x[:, :-p]
            features = features[:-p] if self.use_features else None
            layers = self.transition_models

        if with_bias:
            if self.use_features:
                bias = self.bias_net(features).squeeze(-1).unsqueeze(0) # shape [1, T, num_nodes]
            else:
                bias = self.bias_param.reshape(1, -1, 1)

            states = states + bias

        for layer in layers:
            if hasattr(self, 'transition_models'):
                states = layer(states, transpose=transpose, features=features)

        if transpose:
            states = torch.cat([states, torch.zeros_like(x[:, x.size(1)-p:]).reshape(x.size(0), p, -1)], dim=1)
        else:
            states = torch.cat([torch.zeros_like(x[:, :p]).reshape(x.size(0), p, -1), states], dim=1)

        return states

    def get_matrix(self, t_start=0, t_end=-1, dtype=torch.float32):

        if self.use_features:
            features = self.features[t_start:t_end]
            T = features.size(0)
        else:
            T = 2
            features = None
        F_t = self.forward(torch.eye(self.num_nodes).reshape(self.num_nodes, 1, self.num_nodes).repeat(1, T, 1),
                features=features, transpose=True, with_bias=False, p=1)[:, :-1, :].squeeze()

        return F_t.to(dtype)


class DGMRFActivation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight_param = torch.nn.parameter.Parameter(
                2*torch.rand(1) - 1.) # U(-1,1)

        # For log-det
        self.n_training_samples = config["n_training_samples"]
        self.last_input = None

    @property
    def activation_weight(self):
        return torch.nn.functional.softplus(self.weight_param)

    def forward(self, x, edge_index, transpose, with_bias):
        self.last_input = x.detach()
        return torch.nn.functional.prelu(x, self.activation_weight)

    def log_det(self):
        # Computes log-det for last input fed to forward
        n_negative = (self.last_input < 0.).sum().to(torch.float32)
        return (1./self.n_training_samples)*n_negative*torch.log(self.activation_weight)


class DGMRFLayer(ptg.nn.MessagePassing):
    def __init__(self, config, graph, vi_layer=False):
        super(DGMRFLayer, self).__init__(aggr="add", node_dim=-1)

        self.edge_index = graph['edge_index']
        self.num_nodes = get_num_nodes(self.edge_index)
        self.degrees = ptg.utils.degree(self.edge_index[0])

        self.alpha1_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)
        self.alpha2_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)

        if config["use_bias"]:
            self.bias = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)
        else:
            self.bias = None

        if config["log_det_method"] == "eigvals":
            assert 'eigvals' in graph, ("Dataset not pre-processed with eigenvalues")
            self.adj_eigvals = graph['eigvals']
            self.eigvals_log_det = True
        elif config["log_det_method"] == "dad":
            assert 'dad_traces' in graph, (
                "Dataset not pre-processed with DAD traces")
            dad_traces = graph['dad_traces']

            # Complete vector to use in power series for log-det-computation
            k_max = len(dad_traces)
            self.power_ks = torch.arange(k_max) + 1
            self.power_series_vec = (dad_traces * torch.pow(-1., (self.power_ks + 1))
                                     ) / self.power_ks
        else:
            assert False, "Unknown log-det method"

        self.log_degrees = torch.log(self.degrees)
        self.sum_log_degrees = torch.sum(self.log_degrees)  # For determinant

        # Degree weighting parameter (can not be fixed for vi)
        self.fixed_gamma = (not vi_layer) and bool(config["fix_gamma"])
        if self.fixed_gamma:
            self.gamma_param = config["gamma_value"] * torch.ones(1)
        else:
            self.gamma_param = torch.nn.parameter.Parameter(2. * torch.rand(1, ) - 1)

        # edge_log_degrees contains log(d_i) of the target node of each edge
        self.edge_log_degrees = self.log_degrees[self.edge_index[1]]
        self.edge_log_degrees_transpose = self.log_degrees[self.edge_index[0]]

    @property
    def degree_power(self):
        if self.fixed_gamma:
            return self.gamma_param
        else:
            # Forcing gamma to be in (0,1)
            return torch.sigmoid(self.gamma_param)

    @property
    def self_weight(self):
        # Forcing alpha1 to be positive is no restriction on the model
        return torch.exp(self.alpha1_param)

    @property
    def neighbor_weight(self):
        # Second parameter is (alpha2 / alpha1)
        return self.self_weight * torch.tanh(self.alpha2_param)


    def weight_self_representation(self, x):
        # Representation of same node weighted with degree (taken to power)
        # x has shape [..., num_nodes]
        return x * torch.exp(self.degree_power * self.log_degrees)

    def forward(self, x, transpose, with_bias):
        # x has shape [num_nodes, n_samples]
        weighted_repr = self.weight_self_representation(x) # shape [..., num_nodes]

        aggr = (self.self_weight * weighted_repr) + (self.neighbor_weight * self.propagate(
            self.edge_index, x=x, transpose=transpose)) # Shape [..., num_nodes]

        if self.bias and with_bias:
            aggr += self.bias

        return aggr

    def message(self, x_j, transpose):
        # x_j are neighbor features
        if transpose:
            log_degrees = self.edge_log_degrees_transpose
        else:
            log_degrees = self.edge_log_degrees

        edge_weights = torch.exp((self.degree_power - 1) * log_degrees)

        weighted_messages = x_j * edge_weights

        return weighted_messages

    def log_det(self):
        if self.eigvals_log_det:
            # Eigenvalue-based method
            eigvals = self.neighbor_weight[0] * self.adj_eigvals + self.self_weight[0]
            agg_contrib = torch.sum(torch.log(torch.abs(eigvals)))  # from (aI+aD^-1A)
            degree_contrib = self.degree_power * self.sum_log_degrees  # From D^gamma
            return agg_contrib + degree_contrib
        else:
            # Power series method, using DAD traces
            alpha_contrib = self.num_nodes * self.alpha1_param
            gamma_contrib = self.degree_power * self.sum_log_degrees
            dad_contrib = torch.sum(self.power_series_vec * \
                                    torch.pow(torch.tanh(self.alpha2_param), self.power_ks))
            return alpha_contrib + gamma_contrib + dad_contrib


class DGMRFLayerMultiChannel(ptg.nn.MessagePassing):
    def __init__(self, config, graph, vi_layer=False, T=1, shared='all', weighted=False):
        super(DGMRFLayerMultiChannel, self).__init__(aggr="add", node_dim=-1)

        self.edge_index = graph['edge_index']
        self.num_nodes = get_num_nodes(self.edge_index)
        self.T = T
        self.shared = shared

        if weighted:
            self.edge_weights = graph.get('edge_weight', torch.ones(self.edge_index.size(1)))
            self.degrees = graph.get('weighted_degrees', ptg.utils.degree(self.edge_index[0], num_nodes=self.num_nodes))
        else:
            self.edge_weights = torch.ones(self.edge_index.size(1))
            self.degrees = ptg.utils.degree(self.edge_index[0], num_nodes=self.num_nodes)

        print(f'edge weights = {self.edge_weights}')


        if self.shared == 'dynamics':
            # same parameters for dynamics transition noise
            self.alpha1_param = torch.nn.parameter.Parameter(2.*torch.rand(2, 1) - 1)
            self.alpha2_param = torch.nn.parameter.Parameter(2.*torch.rand(2, 1) - 1)
        elif self.shared == 'all':
            # same parameters for all time steps
            self.alpha1_param = torch.nn.parameter.Parameter(2. * torch.rand(1, 1) - 1)
            self.alpha2_param = torch.nn.parameter.Parameter(2. * torch.rand(1, 1) - 1)
        else:
            # different parameters for all time steps
            self.alpha1_param = torch.nn.parameter.Parameter(2.*torch.rand(self.T, 1) - 1)
            self.alpha2_param = torch.nn.parameter.Parameter(2.*torch.rand(self.T, 1) - 1)

        if config["use_bias"]:
            if self.shared == 'dynamics':
                # spatial bias is zero for t > 0
                self.bias_param = torch.nn.parameter.Parameter(2.*torch.rand(1,) - 1)
            else:
                self.bias_param = torch.nn.parameter.Parameter(2.*torch.rand(self.alpha1_param.size()) - 1)
        else:
            self.bias_param = None

        if config["log_det_method"] == "eigvals":
            assert 'eigvals' in graph, (
                "Dataset not pre-processed with eigenvalues")
            self.adj_eigvals = graph.get('weighted_eigvals', graph['eigvals']) if weighted else graph['eigvals']
            self.eigvals_log_det = True
        elif config["log_det_method"] == "dad":
            assert 'dad_traces' in graph, (
                "Dataset not pre-processed with DAD traces")
            dad_traces = graph.get('weighted_dad_traces', graph['dad_traces']) if weighted else graph['dad_traces']

            # Complete vector to use in power series for log-det-computation
            k_max = len(dad_traces)
            self.power_ks = torch.arange(k_max) + 1 # [k_max]
            self.power_series_vec = (dad_traces * torch.pow(-1., (self.power_ks + 1))
                                     ) / self.power_ks # [k_max]
        else:
            assert False, "Unknown log-det method"

        self.log_degrees = torch.log(self.degrees).unsqueeze(0)
        self.sum_log_degrees = torch.sum(self.log_degrees)  # For determinant

        # Degree weighting parameter (can not be fixed for vi)
        self.fixed_gamma = (not vi_layer) and bool(config["fix_gamma"])
        if self.fixed_gamma:
            self.gamma_param = config["gamma_value"] * torch.ones(self.T, 1)
        else:
            self.gamma_param = torch.nn.parameter.Parameter(2. * torch.rand(self.alpha1_param.size()) - 1)

        # edge_log_degrees contains log(d_i) of the target node of each edge
        self.edge_log_degrees = self.log_degrees[:, self.edge_index[1]]
        self.edge_log_degrees_transpose = self.log_degrees[:, self.edge_index[0]]

    @property
    def gamma(self):
        if self.shared == 'dynamics':
            return torch.cat([self.gamma_param[0].unsqueeze(0),
                              self.gamma_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
        else:
            return self.gamma_param

    @property
    def alpha1(self):
        if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
            return torch.cat([self.alpha1_param[0].unsqueeze(0),
                              self.alpha1_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
        else:
            return self.alpha1_param

    @property
    def alpha2(self):
        if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
            return torch.cat([self.alpha2_param[0].unsqueeze(0),
                              self.alpha2_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
        else:
            return self.alpha1_param

    @property
    def bias(self):
        if self.shared == 'dynamics':
            # return torch.cat([self.bias_param[0].unsqueeze(0),
            #                   self.bias_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
            return torch.cat([self.bias_param.unsqueeze(0), torch.zeros(self.T - 1, 1)], dim=0)
        else:
            return self.bias_param

    @property
    def degree_power(self):
        if self.fixed_gamma:
            return self.gamma_param
        else:
            # Forcing gamma to be in (0,1)
            return torch.sigmoid(self.gamma)

    @property
    def self_weight(self):
        # Forcing alpha1 to be positive is no restriction on the model
        return torch.exp(self.alpha1)

    @property
    def neighbor_weight(self):
        # Second parameter is (alpha2 / alpha1)
        return self.self_weight * torch.tanh(self.alpha2)

    def weight_self_representation(self, x):
        # Representation of same node weighted with degree (taken to power)
        # x has shape [..., num_nodes]
        # [..., T, num_nodes] * [T, 1] * [1, num_nodes]
        return x * torch.exp(self.degree_power * self.log_degrees)

    def forward(self, x, transpose, with_bias):
        # x has shape [..., T, num_nodes]
        weighted_repr = self.weight_self_representation(x) # shape [..., T, num_nodes]

        # [T, 1] * [..., T, num_nodes] + ([channels, 1] * [..., T, num_nodes]
        aggr = (self.self_weight * weighted_repr) + (self.neighbor_weight * self.propagate(
            self.edge_index, x=x, transpose=transpose)) # Shape [..., num_nodes]

        if with_bias and self.bias_param is not None:
            aggr += self.bias

        return aggr

    def message(self, x_j, transpose):
        # x_j are neighbor features of size [..., T, num_edges]
        if transpose:
            log_degrees = self.edge_log_degrees_transpose
        else:
            log_degrees = self.edge_log_degrees

        # shape [channels, num_edges]
        edge_weights = torch.exp((self.degree_power - 1) * log_degrees) * self.edge_weights.unsqueeze(0)

        weighted_messages = x_j * edge_weights

        return weighted_messages

    def log_det(self):
        if self.eigvals_log_det:
            # Eigenvalue-based method
            # [T, 1] * [1, n_eigvals] + [T, 1]
            eigvals = self.neighbor_weight * self.adj_eigvals.unsqueeze(0) + self.self_weight # [T, n_eigvals]
            agg_contrib = torch.log(torch.abs(eigvals)).sum(1).unsqueeze(1)  # [T, 1]
            degree_contrib = self.degree_power * self.sum_log_degrees  # From D^gamma
            channel_log_dets = agg_contrib + degree_contrib
        else:
            # Power series method, using DAD traces
            alpha_contrib = self.num_nodes * self.alpha1 # [channels, 1]
            gamma_contrib = self.degree_power * self.sum_log_degrees # [T, 1]
            dad_contrib = (self.power_series_vec.unsqueeze(0) * \
                                    torch.pow(torch.tanh(self.alpha2), self.power_ks)).sum(1).unsqueeze(1)
            channel_log_dets = alpha_contrib + gamma_contrib + dad_contrib

        if self.shared == 'all':
            # same log det for all time points
            return channel_log_dets.sum() * self.T
        else:
            return channel_log_dets.sum()




class DGMRF(torch.nn.Module):
    def __init__(self, config, graph, T=1, shared='all', weighted=False):
        super(DGMRF, self).__init__()

        self.edge_index = graph['edge_index']
        self.num_nodes = get_num_nodes(self.edge_index)
        self.T = T

        layer_list = []
        for layer_i in range(config["n_layers"]):
            layer_list.append(DGMRFLayerMultiChannel(config, graph, vi_layer=False, T=T,
                                                     shared=shared, weighted=weighted))

        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, x, transpose=False, with_bias=True, **kwargs):

        if transpose:
            # Transpose operation means reverse layer order
            layer_iter = reversed(self.layers)
        else:
            layer_iter = self.layers

        for layer in layer_iter:
            x = layer(x, transpose, with_bias)

        return x

    def log_det(self):
        return sum([layer.log_det() for layer in self.layers])

    def get_matrices(self):
        input = torch.eye(self.num_nodes).reshape(self.num_nodes, 1, self.num_nodes).repeat(1, self.T, 1)
        print('get_matrices', input.size())
        all_Q = self.forward(input, transpose=True, with_bias=False)

        Q_0 = all_Q[:, 0].squeeze()
        Q_t = all_Q[:, 1].squeeze()

        return Q_0, Q_t

    def get_inv_matrices(self):
        Q_0, Q_t = self.get_matrices()

        Q_0_inv = torch.inverse(Q_0)
        Q_t_inv = torch.inverse(Q_t)

        return Q_0_inv, Q_t_inv


class DiagonalModel(torch.nn.Module):
    def __init__(self, config, graph, T=1, shared='all', **kwargs):
        super(DiagonalModel, self).__init__()

        self.edge_index = graph['edge_index']
        self.num_nodes = get_num_nodes(self.edge_index)
        self.T = T

        self.shared = shared

        self.precision_param = Parameter(torch.rand(2, 1) if self.shared == 'dynamics' else torch.rand(1))
        self.bias_param = Parameter(torch.rand(2, 1) * 2 - 1 if self.shared == 'dynamics' else torch.rand(1) * 2 - 1)

    @property
    def precision(self):
        if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
            return torch.cat([self.precision_param[0], self.precision_param[1].repeat(self.T - 1)]).unsqueeze(-1)
        else:
            # use same parameters for all time steps 0, 1, ..., T
            return self.precision_param.repeat(self.T).unsqueeze(-1)

    @property
    def bias(self):
        if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
            return torch.cat([self.bias_param[0], self.bias_param[1].repeat(self.T - 1)]).unsqueeze(-1)
        else:
            # use same parameters for all time steps 0, 1, ..., T
            return self.bias_param.repeat(self.T).unsqueeze(-1)

    def forward(self, x, with_bias=True, **kwargs):
        # x has shape [..., T, num_nodes]

        x = self.precision * x
        if with_bias:
            x = x + self.bias

        return x

    def log_det(self):
        return self.num_nodes * self.precision.sum()

    def get_matrices(self):

        Q_0 = torch.eye(self.num_nodes) * self.precision[0]
        Q_t = torch.eye(self.num_nodes) * self.precision[-1]

        return Q_0, Q_t

    def get_inv_matrices(self):

        Q_0_inv = torch.eye(self.num_nodes) / self.precision[0]
        Q_t_inv = torch.eye(self.num_nodes) / self.precision[-1]

        return Q_0_inv, Q_t_inv



# Solve Q_tilde x = rhs using Conjugate Gradient
def cg_solve(rhs, dgmrf, mask, T, config, noise_var=None, features=None,
             verbose=False, initial_guess=None, return_info=False, preconditioner=None, regularizer=0):
    # rhs has shape [n_batch, T * n_nodes]

    n_batch = rhs.size(0)
    N = mask.numel()

    # CG requires more precision for numerical stability
    rhs = rhs.to(torch.float64)

    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)
    else:
        initial_guess = initial_guess.to(torch.float64)

    b = rhs + regularizer * initial_guess

    if noise_var is not None:
        noise_var = noise_var.to(torch.float64)

    # Batch linear operator
    def Q_tilde_batched(x):
        # x has shape (n_batch, T * n_nodes, 1)

        x = x.reshape(n_batch, T, -1)

        Gx = dgmrf(x, with_bias=False)
        GtGx = dgmrf(Gx, transpose=True, with_bias=False) # has shape [n_batch, T, n_nodes]

        if noise_var is not None:
            res = GtGx + (mask.to(torch.float64) / noise_var).view(1, T, -1) * x
        else:
            res = GtGx

        res = res + regularizer * x # regularization
        res = res.view(n_batch, N, 1)

        return res

    if config["use_features"] and features is not None:
        # features has shape [T, n_nodes, n_features]
        masked_features = features.to(torch.float64) * mask.to(torch.float64).view(N, 1) # shape [n_nodes * T, n_features]
        masked_features_cov = masked_features.transpose(0,1)@masked_features # shape [n_features, n_features]

        def Q_tilde_batched_with_features(x):
            # x has shape (n_batch, N + n_features, 1)
            node_x = x[:,:N]
            coeff_x = x[:,N:]

            top_res1 = Q_tilde_batched(node_x) # shape [n_batch, N, 1]

            top_res2 = (masked_features @ coeff_x) / noise_var # shape [n_batch, N, 1]

            bot_res1 = (masked_features.transpose(0,1) @ node_x) / noise_var # shape [n_batch, n_features, 1]
            bot_res2 = (masked_features_cov @ coeff_x) / noise_var +\
                    (config["coeff_inv_std"]**2)*coeff_x # shape [n_batch, n_features, 1]

            res = torch.cat([top_res1 + top_res2, bot_res1 + bot_res2], dim=1) # shape [n_batch, N + n_features, 1]

            return res

        Q_tilde_func = Q_tilde_batched_with_features
    else:
        Q_tilde_func = Q_tilde_batched

    solution, cg_info = cg_batch.cg_batch(Q_tilde_func, b.unsqueeze(-1), X0=initial_guess.unsqueeze(-1),
                                          rtol=config["inference_rtol"], maxiter=config["max_cg_iter"], verbose=verbose)

    residuals = rhs + regularizer * solution.squeeze(-1) - Q_tilde_func(solution).squeeze(-1)
    res_norm = torch.linalg.norm(residuals, dim=1)
    cg_info["res_norm"] = res_norm

    if verbose:
        print("CG finished in {} iterations, solution optimal: {}".format(
            cg_info["niter"], cg_info["optimal"]))

    if return_info:
        return solution.to(torch.float32), cg_info
    else:
        return solution.to(torch.float32)

def get_bias(dgmrf, input_shape):
    bias = dgmrf(torch.zeros(input_shape))
    return bias


@torch.no_grad()
def sample_posterior(n_samples, dgmrf, data, mask, config, noise_var, initial_guess, return_info=False, features=None,
                     preconditioner=None):
    # Construct RHS using Papandeous and Yuille method
    bias = get_bias(dgmrf, data.size())
    T, num_nodes = data.shape[1:]
    std_gauss1 = torch.randn(n_samples, T, num_nodes)
    std_gauss2 = torch.randn(n_samples, T * num_nodes)

    # G^T @ (z_1 - b)
    rhs_sample1 = dgmrf(std_gauss1 - bias, transpose=True, with_bias=False).reshape(n_samples, -1)

    # H^T @ R^{-1} @ (y + z_2)
    rhs_sample2 = (mask.to(torch.float32) * data.to(torch.float32).reshape(-1, T * num_nodes)
                   + mask.to(torch.float32) * noise_var.sqrt() * std_gauss2) / noise_var

    rhs_sample = rhs_sample1 + rhs_sample2

    initial_guess = initial_guess.reshape(1, -1)

    if config["use_features"] and features is not None:
        # Change rhs to also sample coefficients
        n_features = features.size(-1)
        std_gauss1_coeff = torch.randn(n_samples, n_features, 1)
        rhs_sample1_coeff = config["coeff_inv_std"] * std_gauss1_coeff

        rhs_sample2_coeff = features.transpose(0,1) @ rhs_sample2.view(n_samples, T * num_nodes, 1)
        rhs_sample_coeff = rhs_sample1_coeff + rhs_sample2_coeff


        rhs_sample = torch.cat((rhs_sample, rhs_sample_coeff.squeeze(-1)), dim=1)

        cg_features = features
        if initial_guess.size(1) < rhs_sample.size(1):
            initial_guess = torch.cat([initial_guess, torch.zeros(1, n_features)], dim=1)
    else:
        cg_features = None

    initial_guess = initial_guess.repeat(n_samples, 1).reshape(rhs_sample.size())
    res_norm = torch.ones(data.size(0)) * float("inf")
    rhs_norm = torch.linalg.norm(rhs_sample, dim=1)
    k = 0
    cg_niter = 0
    regularizer = config.get('cg_regularizer', 0)
    while (res_norm > config.get('outer_rtol', 1e-7) * rhs_norm).any() and (k < config.get('max_outer_iter', 100)):
        samples, cg_info = cg_solve(rhs_sample, dgmrf, mask, data.size(1), config,
                                      noise_var=noise_var, verbose=False, return_info=True,
                                      initial_guess=initial_guess, features=cg_features,
                                      preconditioner=preconditioner, regularizer=regularizer)

        k = k + 1
        cg_niter += cg_info["niter"]
        res_norm = cg_info["res_norm"]
        print(f'relative residual norm after outer iteration {k} = {res_norm / rhs_norm}')
        initial_guess = samples.squeeze(-1)
        
        if k % 10 == 0:
            regularizer = max(config.get('cg_gamma', 0.1) * regularizer, config.get('cg_min_regularizer', 1e-7))

    cg_info["niter"] = cg_niter

    if config["use_features"] and features is not None:
        samples_x = samples[:, :mask.numel()]
        samples_beta = samples[:, mask.numel():]

        print(samples_x.size(), features.size(), samples_beta.size())

        samples = samples_x + features @ samples_beta
        print(samples.size())

    if return_info:
        return samples, cg_info
    else:
        return samples


@torch.no_grad()
def sample_prior(n_samples, dgmrf, data_shape, config, features=None):

    bias = get_bias(dgmrf, data_shape)
    std_gauss = torch.randn(n_samples, data_shape[1], data_shape[2])

    rhs_sample = dgmrf(std_gauss - bias, transpose=True, with_bias=False) # G^T @ (z - b)

    if config["use_features"] and features is not None:
        # Also sample coefficients
        n_features = features.size(-1)
        std_gauss_coeff = torch.randn(n_samples, n_features, 1)
        rhs_sample_coeff = config["coeff_inv_std"] * std_gauss_coeff

        rhs_sample = torch.cat((rhs_sample, rhs_sample_coeff.squeeze(-1)), dim=1)

    initial_guess = torch.zeros(rhs_sample.size())
    res_norm = torch.ones(data_shape[0]) * float("inf")
    rhs_norm = torch.linalg.norm(rhs_sample, dim=1)
    k = 0
    cg_niter = 0
    regularizer = config.get('cg_regularizer', 0)
    while (res_norm > config.get('outer_rtol', 1e-7) * rhs_norm).any() and (k < config.get('max_outer_iter', 100)):
        samples, cg_info = cg_solve(rhs_sample, dgmrf, mask, data_shape[1], config,
                                    noise_var=0, verbose=False, return_info=True,
                                    initial_guess=initial_guess, features=features,
                                    regularizer=regularizer)

        k = k + 1
        cg_niter += cg_info["niter"]
        res_norm = cg_info["res_norm"]
        print(f'relative residual norm after outer iteration {k} = {res_norm / rhs_norm}')
        initial_guess = samples.squeeze(-1)
        regularizer = 0.1 * regularizer

    cg_info["niter"] = cg_niter

    if return_info:
        return samples, cg_info
    else:
        return samples


@torch.no_grad()
def posterior_mean(dgmrf, data, mask, config, noise_var, initial_guess, verbose=False, preconditioner=None, features=None,
                   return_info=False):
    # data has shape [n_batch, T, num_nodes]
    bias = get_bias(dgmrf, data.size())
    eta = -1. * dgmrf(bias, transpose=True, with_bias=False)  # -G^T @ b

    masked_y = mask.to(torch.float32).reshape(1, *data.shape[1:]) * data.to(
        torch.float32)  # H^T @ y (has shape [1, T, num_nodes])
    mean_rhs = eta + masked_y / noise_var  # eta + H^T @ R^{-1} @ y

    mean_rhs = mean_rhs.reshape(data.size(0), -1)
    initial_guess = initial_guess.reshape(mean_rhs.size())

    if config["use_features"] and features is not None:
        rhs_append = (features.transpose(0, 1) @ masked_y.reshape(-1, 1)) / noise_var
        mean_rhs = torch.cat([mean_rhs, rhs_append.view(1, -1).repeat(mean_rhs.size(0), 1)],
                             dim=1)  # shape [n_batch, T * num_nodes + n_features]

        if initial_guess.size(1) < mean_rhs.size(1):
            initial_guess = torch.cat([initial_guess, torch.zeros(data.size(0), features.size(-1))], dim=1)
        cg_features = features

    else:
        cg_features = None
    
    res_norm = torch.ones(data.size(0)) * float("inf")
    rhs_norm = torch.linalg.norm(mean_rhs, dim=1)
    k = 0
    cg_niter = 0
    regularizer = config.get('cg_regularizer', 0)
    
    while (res_norm > config.get('outer_rtol', 1e-7) * rhs_norm).any() and (k < config.get('max_outer_iter', 100)):
        post_mean, cg_info = cg_solve(mean_rhs, dgmrf, mask, data.size(1), config,
                                      noise_var=noise_var, verbose=verbose, return_info=True,
                                      initial_guess=initial_guess, features=cg_features,
                                      preconditioner=preconditioner, regularizer=regularizer)

        k = k + 1
        cg_niter += cg_info["niter"]
        res_norm = cg_info["res_norm"]
        print(f'relative residual norm after outer iteration {k} = {res_norm / rhs_norm}')
        initial_guess = post_mean.squeeze(-1)
        
        if k % 10 == 0:
            regularizer = max(config.get('cg_gamma', 0.1) * regularizer, config.get('cg_min_regualizer', 1e-7))

    post_mean = post_mean[0]
    cg_info["niter"] = cg_niter

    if config["use_features"] and features is not None:
        post_mean_x = post_mean[:mask.numel()]
        post_mean_beta = post_mean[mask.numel():]

        post_mean = post_mean_x + features @ post_mean_beta

        print(f'beta = {post_mean_beta}')
    
    if return_info:
        return post_mean, cg_info
    else:
        return post_mean

@torch.no_grad()
def posterior_inference(dgmrf, data, mask, config, noise_var, initial_guess, features=None,
                        verbose=False, return_time=False, preconditioner=None):
    # data has shape [n_batch, T, num_nodes]

    print(f'compute posterior mean')

    start = timer()
    post_mean, cg_info = posterior_mean(dgmrf, data, mask, config, noise_var, initial_guess, return_info=True, features=features,
                                        preconditioner=preconditioner, verbose=verbose)
    end = timer()
    time_per_iter = (end - start) / cg_info["niter"]
    mean_niter = cg_info["niter"]

    print(f'draw samples from posterior')

    posterior_samples_list = []
    cur_post_samples = 0
    niter_list = []
    while cur_post_samples < config["n_post_samples"]:
        samples, cg_info = sample_posterior(config["n_training_samples"], dgmrf, data, mask, config,
                                            noise_var, initial_guess, return_info=True, features=features,
                                            preconditioner=preconditioner)
        posterior_samples_list.append(samples)
        cur_post_samples += config["n_training_samples"]
        niter_list.append(cg_info["niter"])

    posterior_samples = torch.cat(posterior_samples_list, dim=0)[:config["n_post_samples"]]


    # MC estimate of variance using known population mean
    post_var_x = torch.mean(torch.pow(posterior_samples - post_mean, 2), dim=0)
    # Posterior std.-dev. for y
    post_std = torch.sqrt(post_var_x + noise_var)

    if return_time:
        return post_mean, post_std, time_per_iter, torch.tensor(niter_list, dtype=torch.float32).mean()
    else:
        return post_mean, post_std, mean_niter


def setup_model(config, device='cpu'):
    dataset_dict = utils.load_dataset(config["dataset"], config["data_dir"], device=device)
    spatial_graph = dataset_dict["spatial_graph"]
    temporal_graph = dataset_dict["temporal_graph"]

    # make sure that edge normals have correct order
    if config['dataset'].startswith('advection') or config['dataset'].startswith('spatiotemporal'):
        normals = torch.stack([utils.get_normal(temporal_graph.pos[u], temporal_graph.pos[v],
                                                max=np.sqrt(spatial_graph.num_nodes) - 1)
                               for u, v in temporal_graph.edge_index.T], dim=0)
        spatial_graph.edge_attr = normals
        temporal_graph.edge_attr = normals

    # load features and normalize them to zero mean and unit variance
    if config.get('use_features', False) or config.get('use_features_dynamics', False):
        features = dataset_dict["covariates"].to(torch.float32)
        features = features - features.mean(0)
        print('features std min', features.std(0).min())
        features = features / features.std(0)
        features = features[:, [0, 3, 4]]
        print(f'use {features.size(1)} features')
    else:
        features = None

    data = dataset_dict["data"].to(torch.float32)
    joint_mask = dataset_dict["masks"].reshape(-1)  # shape [T * num_nodes]

    N = dataset_dict["masks"].numel()
    T = dataset_dict["masks"].size(0)

    if not config.get('final', False):
        # exclude all test data for training and validation runs
        trainval_mask = torch.logical_not(dataset_dict["test_masks"].reshape(-1))
        data = data[trainval_mask[joint_mask]]
        joint_mask = torch.logical_and(joint_mask, trainval_mask)

    initial_guess = torch.ones(N) * data.mean()

    model = SpatiotemporalInference(config, initial_guess, data, joint_mask,
                                    spatial_graph.to_dict(), temporal_graph.to_dict(),
                                    T=T, gt=dataset_dict.get('gt', None),
                                    data_mean=dataset_dict.get('data_mean', 0),
                                    data_std=dataset_dict.get('data_std', 1),
                                    features=features,
                                    true_post_mean=dataset_dict.get("true_posterior_mean", None),
                                    true_post_std=dataset_dict.get("true_posterior_std", None))

    return model


