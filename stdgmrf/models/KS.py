import torch
from torch import nn
import torch_geometric as ptg
import pytorch_lightning as pl
import math
from torch_geometric.nn import MessagePassing
import torch_geometric as tg
import torch.nn.functional as F
from stdgmrf import utils



class KalmanSmoother:

    def __init__(self, initial_mean, initial_cov, transition_model, transition_cov,
                 observation_models, observation_noise):

        self.initial_mean = initial_mean # size [batch, state]
        self.initial_cov = initial_cov # size [batch, state, state]
        self.F = transition_model # size [batch, state, state]
        self.Q = transition_cov # size [batch, state, state]
        self.H = observation_models # list of tensors with size [batch, data_t, state]
        # self.R = observation_cov # size [batch, data, data]
        self.sigma_obs = observation_noise # size [batch]

        self.T = len(self.H)
        self.batch_dim, self.state_dim = self.initial_mean.size()

    def update_H(self, observation_models):
        self.H = observation_models
        self.T = len(self.H)


    def EM(self, data, iterations, update=['alpha', 'F', 'Q', 'R', 'mu', 'Sigma'], eps=1e-3):

        T = data.shape[1]
        i = 0
        delta = float("inf")

        mean, cov, cov_lagged = self.smoother(data)

        while i < iterations and delta > eps:
            print(f'iteration {i}')
            # E-step

            old_mean = mean
            delta_old = delta

            for var in update:

                assert not torch.isnan(cov_lagged).any()
                assert not torch.isnan(cov).any()
                assert not torch.isnan(mean).any()

                assert not torch.isnan(cov_lagged.sum(1)).any()

                # M-step
                A = cov_lagged.sum(1) + (mean[:, :-1].unsqueeze(-1) @ mean[:, 1:].unsqueeze(2)).sum(1) # (batch, state, state)
                B = cov[:, :-1].sum(1) + (mean[:, :-1].unsqueeze(-1) @ mean[:, :-1].unsqueeze(2)).sum(1) # (batch, state, state)
                C = cov[:, 1:].sum(1) + (mean[:, 1:].unsqueeze(-1) @ mean[:, 1:].unsqueeze(2)).sum(1) # (batch, state, state)

                assert not torch.isnan(A).any()
                assert not torch.isnan(B).any()
                assert not torch.isnan(C).any()

                old_F = self.F
                old_Q = self.Q
                old_mu0 = self.initial_mean


                if var == 'alpha':
                    traceA = torch.diagonal(A, dim1=-1, dim2=-2).sum(-1)
                    traceB = torch.diagonal(B, dim1=-1, dim2=-2).sum(-1)
                    alpha = traceA / traceB
                    
                    self.F = alpha.reshape(-1, 1, 1) * torch.eye(self.state_dim).unsqueeze(0).repeat(self.batch_dim, 1, 1)
                elif var == 'F':
                    B_inv = torch.inverse(B)
                    self.F = A @ B_inv

                elif var == 'Q':
                    self.Q = (C - self.F @ A - A @ self.F.transpose(1, 2) + self.F @ B @ self.F.transpose(1, 2)) / (T - 1)
                    assert not torch.isnan(self.Q).any()

                elif var == 'mean': self.initial_mean = mean[:, 0]


                mean, cov, cov_lagged = self.smoother(data)

            delta = torch.pow(mean - old_mean, 2).mean()
            print(f'delta = {delta}')

            if delta > delta_old:
                # recover previous parameter settings
                self.F = old_F
                self.Q = old_Q
                self.initial_mean = old_mu0
                break


            i = i + 1

    def smoother(self, data):

        T = data.shape[1]

        mean_smoothed = torch.zeros(self.batch_dim, T, self.state_dim).type_as(data)
        cov_smoothed = torch.zeros(self.batch_dim, T, self.state_dim, self.state_dim).type_as(data)
        cov_lagged = torch.zeros(self.batch_dim, T - 1, self.state_dim, self.state_dim).type_as(data)

        transition = self.F

        mean_predicted, cov_predicted, mean_filtered, cov_filtered, last_K = self.filter(data)

        for t in range(T):
            # index to be updated
            tidx = T - t - 1

            if t == 0:
                # last filtered state is same as smoothed state
                mean_s = mean_filtered[:, tidx]
                cov_s = cov_filtered[:, tidx]
                cov_lag1 = (torch.eye(self.state_dim) - last_K @ self.H[tidx]) @ transition @ cov_filtered[:, tidx - 1]

                C_prev = cov_filtered[:, tidx - 1] @ transition.transpose(1, 2) @ torch.inverse(cov_predicted[:, tidx])
            else:
                # C = P_k^f A_k^T (P_{k+1}^p){-1}
                C = C_prev

                # mean update: mu_k^s = mu_k^f + G_k (mu_{k+1}^s - mu_{k+1}^p)
                residual_mean = mean_s - mean_predicted[:, tidx + 1]
                mean_s = mean_filtered[:, tidx] + (C @ residual_mean.unsqueeze(-1)).squeeze(-1)

                # cov update: cov_k^s = cov_k^f + G_k (cov_{k+1}^s - cov_{k+1}^p) G_k^T
                residual_cov = cov_s - cov_predicted[:, tidx + 1]
                cov_s = cov_filtered[:, tidx] + C @ residual_cov @ C.transpose(1, 2)

                if tidx > 0:
                    assert not torch.isnan(cov_predicted[:, tidx]).any()
                    assert not torch.isnan(torch.inverse(cov_predicted[:, tidx])).any()
                    C_prev = cov_filtered[:, tidx - 1] @ transition.transpose(1, 2) @ torch.inverse(cov_predicted[:, tidx])

                    cov_lag1 = C_prev @ cov_s

            mean_smoothed[:, tidx] = mean_s
            cov_smoothed[:, tidx] = cov_s
            if tidx > 0:
                cov_lagged[:, tidx - 1] = cov_lag1

        return mean_smoothed, cov_smoothed, cov_lagged

    def filter(self, data):
        T = data.shape[1]

        mean_filtered = torch.zeros(self.batch_dim, T, self.state_dim)
        mean_predicted = torch.zeros(self.batch_dim, T, self.state_dim)
        cov_filtered = torch.zeros(self.batch_dim, T, self.state_dim, self.state_dim)
        cov_predicted = torch.zeros(self.batch_dim, T, self.state_dim, self.state_dim)

        mean_p = self.initial_mean
        cov_p = self.initial_cov
        transition = self.F

        for t in range(T):

            if t > 0:
                # forecast
                mean_p = (transition @ mean_f.unsqueeze(-1)).squeeze(-1)
                cov_p = (transition @ cov_f @ transition.transpose(1, 2)) + self.Q

            # Kalman gain
            K = cov_p @ self.H[t].transpose(1, 2) @ torch.inverse((self.H[t] @
                        cov_p @ self.H[t].transpose(1, 2)) + self.sigma_obs)

            # analysis
            residual = self.H[t] @ (data[:, t].unsqueeze(-1) - mean_p.unsqueeze(-1))
            innovation = K @ residual
            mean_f = mean_p + innovation.squeeze(-1)

            diff = (torch.eye(self.state_dim) - K @ self.H[t])
            cov_f = diff @ cov_p @ diff.transpose(1, 2) + self.sigma_obs * K @ K.transpose(1, 2)

            mean_filtered[:, t] = mean_f
            mean_predicted[:, t] = mean_p
            cov_filtered[:, t] = cov_f
            cov_predicted[:, t] = cov_p

        return mean_predicted, cov_predicted, mean_filtered, cov_filtered, K

    def prior(self, T):

        mean = torch.zeros(self.batch_dim, T, self.state_dim)
        cov = torch.zeros(self.batch_dim, T, self.state_dim, self.state_dim)

        mean[:, 0] = self.initial_mean
        cov[:, 0] = self.initial_cov
        transition = self.F

        for t in range(1, T):

            mean[:, t] = (transition @ mean[:, t-1].unsqueeze(-1)).squeeze(-1)
            cov[:, t] = (transition @ cov[:, t-1] @ transition.transpose(1, 2)) + self.Q

        return mean, cov


class EnsembleKalmanSmoother:

    def __init__(self, ensemble_size, initial_mean, initial_cov_factor, transition_model, transition_cov_factor,
                 observation_models, observation_noise):

        self.ensemble_size = ensemble_size

        self.initial_mean = initial_mean  # size [state]
        self.initial_cov_factor = initial_cov_factor  # size [state, state]
        self.transition_model = transition_model  # function
        self.transition_cov_factor = transition_cov_factor  # size [state, state]
        self.H = observation_models  # list of tensors with size [data_t, state]
        self.sigma_obs = observation_noise

        self.T = len(self.H)
        self.state_dim = self.initial_mean.size(0)


    def update_H(self, observation_models):
        self.H = observation_models
        self.T = len(self.H)

    def smoother(self, data):

        T = data.shape[0]

        state_ensembles_f, state_ensembles_p = self.filter(data)

        state_ensembles_s = torch.zeros(T, self.state_dim, self.ensemble_size)

        for t in range(T):
            # index to be updated
            tidx = T - t - 1

            if t == 0:
                # last filtered state is same as smoothed state
                state_ensembles_s[tidx] = state_ensembles_f[tidx]
            else:
                residuals = state_ensembles_s[tidx + 1] - state_ensembles_p[tidx + 1]

                # ensemple covariances
                cov_p = self.ensemble_cov(state_ensembles_p[tidx + 1])
                cov_f = self.ensemble_cov(state_ensembles_f[tidx])

                # apply inverse, or alternatively solve with CG or Cholesky
                residuals = torch.inverse(cov_p) @ residuals # shape [state_dim, ensemble_size]

                cov_lagged = self.ensemble_cov(state_ensembles_f[tidx], state_ensembles_p[tidx + 1])
                state_ensembles_s[tidx] = state_ensembles_f[tidx] + cov_lagged @ residuals


        return state_ensembles_f, state_ensembles_p, state_ensembles_s


    def filter(self, data):
        # data has shape [T, num_obs]

        T = data.shape[0]

        state_ensembles_p = torch.zeros(T, self.state_dim, self.ensemble_size)
        state_ensembles_f = torch.zeros(T, self.state_dim, self.ensemble_size)

        state_ensembles_p[0] = self.initial_mean.view(self.state_dim, 1) + \
                         self.initial_cov_factor @ torch.randn(self.state_dim, self.ensemble_size)
        data_ensembles = data.unsqueeze(-1) + self.sigma_obs * torch.randn(T, data.size(1), self.ensemble_size)

        for t in range(T):

            if t > 0:
                # forecast
                model_error = self.transition_cov_factor @ torch.randn(self.state_dim, self.ensemble_size)
                state_ensembles_p[t] = self.transition_model(state_ensembles_f[t-1]) + model_error

            if torch.any(torch.isnan(state_ensembles_p)):
                print(f'found NaNs in predicted ensemble at time t={t}')

            # ensemble covariance
            cov_p = self.ensemble_cov(state_ensembles_p[t])

            # Kalman gain
            K = cov_p @ self.H[t].transpose(0, 1) @ torch.inverse((self.H[t] @
                                                                   cov_p @ self.H[t].transpose(0, 1)) + self.sigma_obs)

            # TODO: implement more efficient version for large number of observations M

            # analysis
            residual = self.H[t] @ (data_ensembles[t] - state_ensembles_p[t])
            innovation = K @ residual

            state_ensembles_f[t] = state_ensembles_p[t] + innovation

            if t==0:
                print(residual)
                print(innovation)


        return state_ensembles_f, state_ensembles_p

    def ensemble_cov(self, ensemble_1, ensemble_2=None):
        A_1 = ensemble_1 - ensemble_1.mean(-1).unsqueeze(-1)  # [state_dim, ensemble_size] or [T, state_dim, ensemble_size]

        if ensemble_2 is None:
            A_2 = A_1
        else:
            A_2 = ensemble_2 - ensemble_2.mean(-1).unsqueeze(
                -1)  # [state_dim, ensemble_size] or [T, state_dim, ensemble_size]
        cov = A_1 @ A_2.transpose(-2, -1) / (self.ensemble_size - 1)

        return cov




class JointKalmanSmoother:

    def __init__(self, initial_mean, initial_precision, transition_graph, transition_precision,
                 observation_model, observation_precision, N, sparse=False):

        self.initial_mean = initial_mean                # size (B * N)
        self.initial_precision = initial_precision      # size (B * N, B * N)
        self.transition = transition_graph
        self.Q_inv = transition_precision               # size (B * N, B * N)
        self.H = observation_model                      # size (B * M, B * N)
        self.R_inv = observation_precision              # size (B * M, B * M)

        self.sparse = sparse
        self.N = N

        self.batch_size, self.data_dim, self.state_dim = self.H.size()



    def prior(self, T):

        mean = []
        omega_tt = []

        F_dense = self.transition.to_dense()

        omega_next_t = - F_dense.transpose(1, 2) @ self.Q_inv

        mean_t = self.initial_mean.reshape(self.batch_size, self.state_dim) # size (B * N)

        assert (self.Q_inv == self.Q_inv.transpose(1, 2)).all()
        assert (self.initial_precision == self.initial_precision.transpose(1, 2)).all()

        L = torch.linalg.cholesky(self.Q_inv)
        A = F_dense.transpose(1, 2) @ L
        A = A @ A.transpose(1, 2)

        assert (A == A.transpose(1, 2)).all()

        for t in range(T):

            mean.append(mean_t)
            mean_t = self.transition(mean_t).reshape(self.batch_size, self.state_dim)

            if t == 0:
                omega_tt.append(self.initial_precision + A)
            elif t == T - 1:
                omega_tt.append(self.Q_inv)
            else:
                omega_tt.append(self.Q_inv + A)

        omega_tt = torch.stack(omega_tt, dim=1) # size (B, T, N, N)
        omega_next_t = omega_next_t.unsqueeze(1).repeat(1, T - 1, 1, 1) # size (B, T-1, N, N)

        joint_mean = torch.cat(mean, dim=1)  # size (B, T * N)

        joint_omega = torch.stack([self.joint_omega(omega_tt[b], omega_next_t[b])
                                   for b in range(self.batch_size)], dim=0) # size (B, T * N, T * N)

        joint_eta = (joint_omega @ joint_mean.unsqueeze(-1)).squeeze(-1)

        return joint_mean, joint_eta, joint_omega

    def joint_prior(self, T):

        mean = []
        D = T * self.state_dim

        mean_t = self.initial_mean.reshape(self.batch_size, self.state_dim)  # size (B * N)
        eta = torch.zeros(self.batch_size, T, self.state_dim)
        eta[:, 0] = (self.initial_precision @ self.initial_mean.reshape(self.batch_size, self.state_dim, 1)).squeeze(-1)

        F_dense = self.transition.to_dense()
        F = F_dense.repeat(T-1, 1, 1)
        F = torch.diag_embed(F.permute(1, 2, 0), offset=-1, dim1=0, dim2=2).reshape(D, D)
        F = (F - torch.eye(T * self.state_dim)).unsqueeze(0)

        L = torch.linalg.cholesky(self.Q_inv)
        L0 = torch.linalg.cholesky(self.initial_precision)
        L = torch.stack([torch.block_diag(L0[b], *L[b].unsqueeze(0).repeat(T-1, 1, 1)) for b in range(F.size(0))], dim=0)

        G = F.transpose(1, 2) @ L
        joint_omega = G @ G.transpose(1, 2)

        for t in range(T):

            mean.append(mean_t)
            mean_t = self.transition(mean_t).reshape(self.batch_size, self.state_dim)

        joint_mean = torch.cat(mean, dim=1)  # size (B, T * N)

        joint_eta = eta.reshape(self.batch_size, T * self.state_dim)

        return joint_mean, joint_eta, joint_omega

    def joint_H(self, T):
        joint_H = torch.stack([torch.block_diag(*self.H[b].unsqueeze(0).repeat(T, 1, 1)) for
                               b in range(self.batch_size)], dim=0)
        return joint_H

    def joint_R_inv(self, T):
        joint_R_inv = torch.stack([torch.block_diag(*self.R_inv[b].unsqueeze(0).repeat(T, 1, 1)) for
                                   b in range(self.batch_size)], dim=0)
        return joint_R_inv

    def smoother(self, data):
        assert data.size(0) == self.batch_size and data.size(2) == self.data_dim

        T = data.size(1)
        B = data.size(0)

        joint_mean, joint_eta, joint_omega = self.joint_prior(T)

        joint_H = self.joint_H(T)
        joint_R_inv = self.joint_R_inv(T)

        R_proj = joint_H.transpose(1, 2) @ joint_R_inv

        updated_omega = joint_omega + R_proj @ joint_H
        updated_eta = joint_eta + (R_proj @ data.reshape(B, T * self.data_dim, 1)).squeeze(-1)

        return joint_mean, joint_eta, joint_omega, updated_eta, updated_omega, joint_R_inv, joint_H

    def sparse_smoother(self, data):

        assert data.size(0) * data.size(2) == self.data_dim # batch_size * n_observations

        T = data.size(1)
        B = data.size(0)
        M = int(self.data_dim / B) # TODO: what if M and N are different among batches?
        mean, omega = self.prior(T)
        eta = torch.sparse.mm(omega, mean.unsqueeze(-1)).squeeze(-1)

        H_t_ = utils.sparse_block_diag_repeat(self.H, T)
        H_t = utils.reverse_nested_block_diag(H_t_, [self.data_dim, self.state_dim], [M, self.N])
        R_t_ = utils.sparse_block_diag_repeat(self.R_inv, T)
        R_t = utils.reverse_nested_block_diag(R_t_, [self.data_dim, self.data_dim], [M, M])

        R_proj = torch.sparse.mm(H_t.transpose(0, 1), R_t)
        omega_new = omega + torch.sparse.mm(R_proj, H_t)
        eta_new = eta + torch.sparse.mm(R_proj, data.reshape(-1, 1)).squeeze(-1)

        return mean, eta, omega, eta_new, omega_new, R_t, H_t


    def joint_omega(self, omega_tt, omega_next_t):

        # TODO: doesn't work for batched omega's !

        omega_tt = torch.block_diag(*omega_tt)
        omega_next_t = torch.diag_embed(omega_next_t.permute(1, 2, 0), offset=1, dim1=0, dim2=2)
        omega_next_t = omega_next_t.reshape(omega_tt.size())
        prev_t = omega_next_t.T

        assert (omega_tt == omega_tt.T).all()
        assert (omega_next_t + prev_t == (omega_next_t + prev_t).T).all()

        joint = omega_tt + omega_next_t + prev_t
        return joint

    def sparse_joint_omega(self, omega_tt, omega_next_t):

        # sizes are (T, B * N, B * N)
        dim0, dim1 = omega_tt[0].size()

        omega_tt = utils.sparse_block_diag(*omega_tt) # size (T * B * N, T * B * N)
        omega_next_t = utils.sparse_block_diag(*omega_next_t, padding=[0, dim0, dim1, 0]) # size (T * B * N, T * B * N)

        joint = omega_tt + omega_next_t + omega_next_t.transpose(0, 1)
        return joint

    def compute_mean(self, joint_omega, eta):

        cov = torch.cholesky_inverse(torch.cholesky(joint_omega))
        mean = (cov @ eta.reshape(self.batch_dim, cov.size(-1), 1)).squeeze()
        return mean


    def EM(self, data, iterations, update=['F', 'Q', 'R', 'mu', 'Sigma'], batch=0):

        T = data.shape[1]

        for i in range(iterations):
            # E-step
            eta, omega_tt, omega_next_t = self.smoother(data)

            omega = torch.stack([self.joint_omega(omega_tt[batch], omega_next_t[batch]) for
                                 batch in range(self.batch_dim)])
            cov = torch.inverse(omega)
            mean = self.compute_mean(omega, eta)

            lag0 = torch.zeros(self.batch_dim, T, self.state_dim, self.state_dim)
            lag1 = torch.zeros(self.batch_dim, T - 1, self.state_dim, self.state_dim)
            rec_error = torch.zeros(self.batch_dim, T, self.data_dim, self.data_dim)

            for t in range(T):
                j = self.state_dim * t
                k = self.state_dim * (t + 1)
                l = self.state_dim * (t + 2)
                lag0[:, t] = cov[:, j:k, j:k] + mean[:, j:k].unsqueeze(-1) @ mean[:, j:k].unsqueeze(-2)
                if t < T - 1:
                    lag1[:, t] = cov[:, j:k, k:l] + mean[:, j:k].unsqueeze(-1) @ mean[:, k:l].unsqueeze(-2)
                error = data[:, t] - (self.H @ mean[:, j:k].unsqueeze(-1)).squeeze()
                rec_error[:, t] = error.unsqueeze(-1) @ error.unsqueeze(-2)

            # M-step
            A = lag1.sum(1)
            B = lag0[:, :-1].sum(1)
            C = lag0[:, 1:].sum(1)
            B_inv = torch.inverse(B)
            D = rec_error.sum(1)
            E = self.H @ lag0.sum(1) @ self.H.transpose(1, 2)

            if 'F' in update: self.F = A @ B_inv
            if 'Q' in update: self.Q_inv = torch.inverse((C - A @ B_inv @ A.transpose(1, 2)) / (T - 1))
            if 'R' in update: self.R_inv = torch.inverse((D + E) / T)
            if 'mean' in update: self.initial_mean = mean[:, :self.state_dim]

            print(f'updated F: {self.F}')



class AdvectionDiffusionTransition(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized advection
    """

    def __init__(self, config, graph):
        super(AdvectionDiffusionTransition, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.edge_index = graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        self.edge_attr = graph['edge_attr'] # normal vectors at cell boundaries

    def forward(self, states, velocity, diffusion, transpose=False, **kwargs):
        # states has shape [num_nodes]
        # velocity_params has shape [2]
        # diffusion_params has shape [1]

        x = states.to(torch.float64)
        out = states.to(torch.float64)

        factor = 1

        for k in range(2):

            factor = factor / (k + 1)

            agg_i, agg_j = self.propagate(self.edge_index, x=x, edge_attr=self.edge_attr.to(torch.float64),
                                          v=velocity.to(torch.float64), diff=diffusion.to(torch.float64))
            if transpose:
                agg_i_T, agg_j_T = self.propagate(self.edge_index_transpose, x=x, edge_attr=self.edge_attr.to(torch.float64),
                                                  v=velocity.to(torch.float64), diff=diffusion.to(torch.float64))
                x = agg_j_T + agg_i
            else:
                x = agg_j + agg_i

            out = out + factor * x

        return out

    def message(self, x_i, x_j, edge_attr, v, diff):
        # construct messages to node i for each edge (j,i)
        # edge_attr has shape [num_edges, 2]
        # v has shape [2]
        # diff has shape [1]

        adv_coef = -0.5 * (edge_attr @ v.unsqueeze(-1)).squeeze(-1) # shape [num_edges]
        msg_i = (adv_coef - diff) * x_i
        msg_j = (adv_coef + diff) * x_j

        return torch.stack([msg_i, msg_j], dim=0)






