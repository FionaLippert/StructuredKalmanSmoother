import torch

class KalmanSmoother:

    def __init__(self, initial_mean, initial_cov, transition_model, transition_cov,
                 observation_model, observation_cov):

        # prior distribution for t=0
        self.initial_mean = initial_mean
        self.initial_cov = initial_cov

        self.N = initial_mean.shape[0]

        self.F = transition_model
        self.Q = transition_cov
        self.H = observation_model
        self.R = observation_cov

    def inference(self, data):

        B, T = data.shape[:2]

        mean_filtered = torch.zeros(B, T, self.N)
        mean_predicted = torch.zeros(B, T, self.N)
        cov_filtered = torch.zeros(B, T, self.N, self.N)
        cov_predicted = torch.zeros(B, T, self.N, self.N)
        mean_smoothed = torch.zeros(B, T, self.N).type_as(data)
        cov_smoothed = torch.zeros(B, T, self.N, self.N).type_as(data)

        mean_p = self.initial_mean.unsqueeze(0).repeat(B, 1)
        cov_p = self.initial_cov.unsqueeze(0).repeat(B, 1, 1)
        transition = self.unsqueeze(0).repeat(B, 1, 1)

        for t in range(T):

            if t > 0:
                # forecast
                mean_p = (transition @ mean_f.unsqueeze(-1)).squeeze(-1)
                cov_p = (transition @ cov_f @ transition.transpose(1, 2)) + self.Q.unsqueeze(0)

            # Kalman gain
            K = cov_p @ self.H.T @ torch.inverse((self.H @ cov_p @ self.H.T) + self.R)

            # analysis
            residual = data[:, t] - (self.H @ mean_p.T).T
            innovation = K @ residual.unsqueeze(-1)
            mean_f = mean_p + innovation.squeeze(-1)

            diff = (torch.eye(self.N) - K @ self.H)
            cov_f = diff @ cov_p @ diff.transpose(1, 2) + K @ self.R @ K.transpose(1, 2)


            mean_filtered[:, t] = mean_f
            mean_predicted[:, t] = mean_p
            cov_filtered[:, t] = cov_f
            cov_predicted[:, t] = cov_p


        for t in range(T):
            # index to be updated
            tidx = T - t - 1

            if t == 0:
                # last filtered state is same as smoothed state
                mean_s = mean_filtered[:, T - 1]
                cov_s = cov_filtered[:, T - 1]
            else:
                # C = P_k^f A_k^T (P_{k+1}^p){-1}
                #transition = transitions[tidx]
                C = cov_filtered[:, tidx] @ transition.transpose(1, 2) @ torch.inverse(cov_predicted[tidx + 1])

                # mean update: mu_k^s = mu_k^f + G_k (mu_{k+1}^s - mu_{k+1}^p)
                residual_mean = mean_s - mean_predicted[tidx + 1]
                mean_s = mean_filtered[tidx] + (C @ residual_mean.unsqueeze(-1)).squeeze(-1)

                # cov update: cov_k^s = cov_k^f + G_k (cov_{k+1}^s - cov_{k+1}^p) G_k^T
                residual_cov = cov_s - cov_predicted[tidx + 1]
                cov_s = cov_filtered[tidx] + C @ residual_cov @ C.transpose(1, 2)

            mean_smoothed[:, tidx] = mean_s
            cov_smoothed[:, tidx] = cov_s

        return mean_smoothed, cov_smoothed

