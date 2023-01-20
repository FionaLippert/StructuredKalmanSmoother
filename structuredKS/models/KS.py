import torch
from torch import nn
import pytorch_lightning as pl



class KalmanSmoother:

    def __init__(self, initial_mean, initial_cov, transition_model, transition_cov,
                 observation_model, observation_cov):

        self.initial_mean = initial_mean
        self.initial_cov = initial_cov
        self.F = transition_model
        self.Q = transition_cov
        self.H = observation_model
        self.R = observation_cov

        self.batch_dim, self.data_dim, self.state_dim = self.H.size()


    def EM(self, data, iterations, update=['F', 'Q', 'R', 'mu', 'Sigma']):

        # TODO: compute terms A, B, C and use them to update parameters
        T = data.shape[1]

        for i in range(iterations):
            # E-step
            mean, cov, cov_lagged = self.smoother(data)

            # M-step
            A = cov_lagged.sum(1) + (mean[:, 1:].unsqueeze(-1) @ mean[:, 1:].unsqueeze(2)).sum(1) # (batch, state, state)
            B = cov[:, :-1].sum(1) + (mean[:, :-1].unsqueeze(-1) @ mean[:, :-1].unsqueeze(2)).sum(1) # (batch, state, state)
            C = cov[:, 1:].sum(1) + (mean[:, 1:].unsqueeze(-1) @ mean[:, 1:].unsqueeze(2)).sum(1) # (batch, state, state)
            B_inv = torch.inverse(B)

            error = data - (self.H.unsqueeze(1) @ mean.unsqueeze(-1)).squeeze() # (batch, time, state)

            F_old = self.F
            Q_old = self.Q
            R_old = self.R
            mean_old = self.initial_mean

            # TODO: use given parameters in update equations if they are not learned

            if 'F' in update: self.F = A @ B_inv
            if 'Q' in update: self.Q = (C - A @ B_inv @ A.transpose(1, 2)) / (T - 1)
            if 'R' in update: self.R = (error.unsqueeze(-1) @ error.unsqueeze(2) +
                      self.H.unsqueeze(1) @ cov @ self.H.transpose(1, 2).unsqueeze(1)).mean(1)
            if 'mean' in update: self.initial_mean = mean[:, 0]
            # self.initial_cov = ?

            print(f'updated Q: {self.Q}')
            # print(f'change in F: {(F_old - self.F).mean()}')
            print(f'change in Q: {(Q_old - self.Q).mean()}')
            # print(f'change in R: {(R_old - self.R).mean()}')
            # print(f'change in mean: {(mean_old - self.initial_mean).mean()}')

    def smoother(self, data):

        T = data.shape[1]

        mean_smoothed = torch.zeros(self.batch_dim, T, self.state_dim).type_as(data)
        cov_smoothed = torch.zeros(self.batch_dim, T, self.state_dim, self.state_dim).type_as(data)
        cov_lagged = torch.zeros(self.batch_dim, T - 1, self.state_dim, self.state_dim).type_as(data)

        transition = self.F #.unsqueeze(0).repeat(B, 1, 1)

        mean_predicted, cov_predicted, mean_filtered, cov_filtered, last_K = self.filter(data)

        for t in range(T):
            # index to be updated
            tidx = T - t - 1

            if t == 0:
                # last filtered state is same as smoothed state
                mean_s = mean_filtered[:, tidx]
                cov_s = cov_filtered[:, tidx]
                cov_lag1 = (torch.eye(self.state_dim) - last_K @ self.H) @ transition @ cov_filtered[:, tidx - 1]

                C_prev = cov_filtered[:, tidx - 1] @ transition.transpose(1, 2) @ torch.inverse(cov_predicted[:, tidx])
            else:
                # C = P_k^f A_k^T (P_{k+1}^p){-1}
                # transition = transitions[tidx]

                #C = cov_filtered[:, tidx] @ transition.transpose(1, 2) @ torch.inverse(cov_predicted[:, tidx + 1])
                C = C_prev

                # mean update: mu_k^s = mu_k^f + G_k (mu_{k+1}^s - mu_{k+1}^p)
                residual_mean = mean_s - mean_predicted[:, tidx + 1]
                mean_s = mean_filtered[:, tidx] + (C @ residual_mean.unsqueeze(-1)).squeeze(-1)

                # cov update: cov_k^s = cov_k^f + G_k (cov_{k+1}^s - cov_{k+1}^p) G_k^T
                residual_cov = cov_s - cov_predicted[:, tidx + 1]
                cov_s = cov_filtered[:, tidx] + C @ residual_cov @ C.transpose(1, 2)

                if tidx > 0:
                    C_prev = cov_filtered[:, tidx - 1] @ transition.transpose(1, 2) @ torch.inverse(cov_predicted[:, tidx])

                    # cov_lag1 = cov_filtered[:, tidx] @ C_prev.transpose(1, 2) + \
                    #            C @ (cov_lag1 - transition @ cov_filtered[:, tidx]) @ C_prev.transpose(1, 2)

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

        mean_p = self.initial_mean #.unsqueeze(0).repeat(B, 1)
        cov_p = self.initial_cov #.unsqueeze(0).repeat(B, 1, 1)
        transition = self.F #.unsqueeze(0).repeat(B, 1, 1)

        for t in range(T):

            if t > 0:
                # forecast
                mean_p = (transition @ mean_f.unsqueeze(-1)).squeeze(-1)
                cov_p = (transition @ cov_f @ transition.transpose(1, 2)) + self.Q #.unsqueeze(0)

            # Kalman gain
            K = cov_p @ self.H.transpose(1, 2) @ torch.inverse((self.H @ cov_p @ self.H.transpose(1, 2)) + self.R)

            # analysis

            #residual = data[:, t] - (self.H @ mean_p.T).T
            residual = data[:, t].unsqueeze(-1) - self.H @ mean_p.unsqueeze(-1)
            innovation = K @ residual #.unsqueeze(-1)
            mean_f = mean_p + innovation.squeeze(-1)

            diff = (torch.eye(self.state_dim) - K @ self.H)
            cov_f = diff @ cov_p  #@ diff.transpose(1, 2) + K @ self.R @ K.transpose(1, 2)

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
            cov[:, t] = (transition @ cov[:, t-1] @ transition.transpose(1, 2)) + self.Q#.unsqueeze(0)

        return mean, cov



class JointKalmanSmoother:

    def __init__(self, initial_mean, initial_precision, transition_model, transition_precision,
                 observation_model, observation_precision):

        # TODO: use @property decorators ?
        self.initial_mean = initial_mean
        self.initial_precision = initial_precision
        self.F = transition_model
        self.Q_inv = transition_precision
        self.H = observation_model
        self.R_inv = observation_precision

        self.batch_dim, self.data_dim, self.state_dim = self.H.size()


    def prior(self, T):

        eta = torch.zeros(self.batch_dim, T, self.state_dim)
        mean = []
        omega_tt = torch.zeros(self.batch_dim, T, self.state_dim, self.state_dim)
        omega_next_t = (- self.F.transpose(1, 2) @ self.Q_inv).unsqueeze(1).repeat(1, T - 1, 1, 1)

        eta[:, 0] = (self.initial_precision @ self.initial_mean.unsqueeze(-1)).squeeze(-1)
        mean_t = self.initial_mean

        # TODO: construct this more efficiently

        for t in range(T):

            mean.append(mean_t)
            mean_t = (self.F @ mean_t.unsqueeze(-1)).squeeze(-1)

            if t == 0:
                omega_tt[:, t] = self.initial_precision + self.F.transpose(1, 2) @ self.Q_inv @ self.F
            elif t == T - 1:
                omega_tt[:, t] = self.Q_inv
            else:
                omega_tt[:, t] = self.Q_inv + self.F.transpose(1, 2) @ self.Q_inv @ self.F

        mean = torch.stack(mean, dim=1)

        return mean, eta, omega_tt, omega_next_t


    def dense_prior(self, T):

        joint_mean = torch.zeros(self.batch_dim, T, self.state_dim)
        joint_mean[:, 0] = self.initial_mean

        joint_cov = torch.zeros(self.batch_dim, T, T, self.state_dim, self.state_dim)
        joint_cov[:, 0, 0] = torch.cholesky_inverse(torch.linalg.cholesky(self.initial_precision))

        Q = torch.inverse(self.Q_inv)

        for t in range(1, T):
            # compute mean for time t
            joint_mean[:, t] = (self.F @ joint_mean[:, t-1].unsqueeze(-1)).squeeze(-1)

            # compute cov elements for time t
            joint_cov[:, t, t] = self.F @ joint_cov[:, t-1, t-1] @ self.F.transpose(1, 2) + Q
            joint_cov[:, :t, t] = joint_cov[:, :t, t-1] @ self.F.transpose(1, 2).unsqueeze(1)
            joint_cov[:, t, :t] = self.F.unsqueeze(1) @ joint_cov[:, t-1, :t]

        return joint_mean, joint_cov



    def smoother(self, data):

        assert data.shape[0] == self.batch_dim

        T = data.size(1)
        mean, eta, omega_tt, omega_next_t = self.prior(T)

        H_t = torch.stack([self.H] * T, dim=1)
        R_t = torch.stack([self.R_inv] * T, dim=1)

        omega_tt_new = omega_tt + H_t.transpose(2, 3) @ R_t @ H_t
        eta_new = eta + (H_t.transpose(2, 3) @ R_t @ data.unsqueeze(-1)).squeeze(-1)

        return mean, eta, omega_tt, omega_next_t, eta_new, omega_tt_new

    def naive_smoother(self, data):

        T = data.size(1)
        B = data.size(0)

        # construct joint prior parameters
        mean, eta, omega_tt, omega_next_t = self.prior(T)
        joint_mean = mean.reshape(B, -1)
        joint_omega = torch.stack([self.joint_omega(omega_tt[b], omega_next_t[b]) for b in range(B)])
        joint_eta = eta.reshape(B, -1)

        # construct joint likelihood parameters
        H = torch.cat([torch.cat([self.H] * T, dim=1)] * T, dim=2)
        R_inv = torch.stack([torch.block_diag(*([self.R_inv[b]] * T)) for b in range(B)])

        # compute joint posterior parameters
        joint_omega_new = joint_omega + H.transpose(1, 2) @ R_inv @ H
        joint_eta_new = joint_eta + (H.transpose(1, 2) @ R_inv @ data.reshape(B, T * self.data_dim, 1)).squeeze(-1)

        return joint_mean, joint_eta, joint_omega, joint_eta_new, joint_omega_new, H, R_inv




    def joint_omega(self, omega_tt, omega_next_t):

        # TODO: doesn't work for batched omega's !

        omega_tt = torch.block_diag(*omega_tt)
        omega_next_t = torch.diag_embed(omega_next_t.permute(1, 2, 0), offset=1, dim1=0, dim2=2)
        omega_next_t = omega_next_t.reshape(omega_tt.size())
        prev_t = omega_next_t.T

        joint = omega_tt + omega_next_t + prev_t
        return joint

    def compute_mean(self, joint_omega, eta):

        cov = torch.inverse(joint_omega)
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


class KSMLE(pl.LightningModule):
    def __init__(self, state_dim, **kwargs):
        super().__init__()
        self.F = nn.parameter.Parameter(torch.eye(state_dim).unsqueeze(0))
        self.lr = kwargs.get('lr', 1e-3)

    # def forward(self, data, ):
    #     T = data.size(1)
    #     self.ks.F = self.F
    #     mean, eta, omega_tt, omega_next_t = self.ks.prior(T)
    #     _, omega_tt_s, omega_next_t_s = self.ks.smoother(data)
    #
    #     joint_mean = mean.reshape(data.size(0), -1)
    #     joint_omega = torch.stack([ks.joint_omega(omega_tt[batch], omega_next_t[batch]) for
    #                          batch in range(data.size(0))])
    #     joint_cov = torch.inverse(joint_omega)
    #     joint_omega_s = torch.stack([ks.joint_omega(omega_tt_s[batch], omega_next_t_s[batch]) for
    #                                batch in range(data.size(0))])
    #     joint_cov_s = torch.inverse(joint_omega_s)
    #
    #     return joint_mean, joint_cov, joint_cov_s


    def training_step(self, batch, batch_idx):

        data = batch['observations']
        T = data.size(1)
        B = data.size(0)

        # set up Kalman smoother
        ks = JointKalmanSmoother(batch['initial_mean'], batch['initial_precision'],
                                      self.F, batch['transition_precision'],
                                      batch['observation_model'], batch['observation_precision'])

        print(f'current F: {self.F}')

        # get prior, posterior and likelihood parameters
        mean, eta, omega_tt, omega_next_t, eta_new, omega_tt_new = ks.smoother(data)


        joint_omega = torch.stack([ks.joint_omega(omega_tt[b], omega_next_t[b]) for b in range(B)])
        joint_omega_new = torch.stack([ks.joint_omega(omega_tt_new[b], omega_next_t[b]) for b in range(B)])

        joint_cov = torch.inverse(joint_omega)
        joint_cov_new = torch.inverse(joint_omega_new)

        # compute (rescaled) negative log-likelihood
        H_t = torch.stack([ks.H] * T, dim=1)
        R_inv_t = torch.stack([ks.R_inv] * T, dim=1)
        error = data.unsqueeze(-1) - (H_t @ mean.unsqueeze(-1)) #.reshape(B, T * ks.data_dim)
        scaled_error = R_inv_t @ error
        projected_error = H_t.transpose(2, 3) @ scaled_error

        scaled_error_2 = (joint_cov_new @ projected_error.reshape(B, T * ks.state_dim, 1)).reshape(B, T, ks.state_dim, 1)
        scaled_error_2 = R_inv_t @ H_t @ scaled_error_2

        rec_loss = error.reshape(B, T, 1, ks.data_dim) @ (scaled_error - scaled_error_2)


        H = torch.cat([torch.cat([ks.H] * T, dim=1)] * T, dim=2)
        R_inv = torch.stack([torch.block_diag(*([ks.R_inv[b]] * T)) for b in range(B)])
        #
        # #error = (data.reshape(B, T * ks.data_dim) - (H @ mean.reshape(B, T * ks.state_dim, 1)).squeeze(-1)).reshape(B, -1)
        # rec_loss = error.reshape(B, 1, T * ks.data_dim) @ (R_inv - R_inv @ H @ joint_cov_new @
        #                                   H.transpose(1, 2) @ R_inv) @ error.unsqueeze(-1)

        # joint_mean, joint_eta, joint_omega, joint_eta_new, joint_omega_new, H, R_inv = ks.naive_smoother(data)
        # joint_cov = torch.inverse(joint_omega)
        logdet = torch.logdet(H @ joint_cov @ H.transpose(1, 2) + torch.inverse(R_inv))

        print(rec_loss.mean(), logdet.mean())

        nll = (rec_loss + logdet).mean() / ks.data_dim # TODO: what if data_dim is different among batches and time steps?


        return nll

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer








