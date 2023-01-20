from structuredKS.datasets import linear
from structuredKS.models.KS import KalmanSmoother, JointKalmanSmoother, KSMLE
from structuredKS import utils
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl



config = {
    'seed': 1234,
    'n_train': 5,
    'grid_size': 16,
    'M': 100,
    'T': 10,
    'dt': 0.2,
    'initial_modes': 5,
    'transition_noise_level': 0.001,
    'observation_noise_level': 0.01,
    'initial_noise_level': 0.01,
    'initial_scale': 1
}

linear_ds = linear.LinearTransport(config, load_data_from='../data')
linear_ds.save('../data')

# test Kalman smoother
ks = KalmanSmoother(linear_ds.initial_mean, torch.cholesky_inverse(torch.linalg.cholesky(linear_ds.initial_precision)) * 100,
                    linear_ds.transition_model, torch.cholesky_inverse(torch.linalg.cholesky(linear_ds.transition_precision)),
                    linear_ds.observation_model, torch.cholesky_inverse(torch.linalg.cholesky(linear_ds.observation_precision)))

mean, cov = ks.prior(config['T'])
torch.save(mean, '../data/prior_mean_marginal.pt')
torch.save(cov, '../data/prior_cov_marginal.pt')

mean, cov, cov_lagged = ks.smoother(linear_ds.observations)
torch.save(mean, '../data/smoothed_mean.pt')
torch.save(cov, '../data/smoothed_cov.pt')

t = 1
print(f'classical smoother: {cov_lagged[0, t]}')
print(f'classical smoother: {cov[0, t]}')


# print(f'true Q: {ks.Q}')
# ks.Q = torch.eye(ks.state_dim, ks.state_dim).unsqueeze(0).repeat(ks.batch_dim, 1, 1)
# ks.EM(linear_ds.observations, 5, update='Q')

# test joint Kalman smoother

#initial_mean = torch.ones_like(linear_ds.initial_mean) * 0.01

ks = JointKalmanSmoother(linear_ds.initial_mean, linear_ds.initial_precision * 0.01,
                    linear_ds.transition_model, linear_ds.transition_precision,
                    linear_ds.observation_model, linear_ds.observation_precision)

_, eta, omega_tt, omega_next_t = ks.prior(config['T'])
joint_mean, joint_cov = ks.dense_prior(config['T'])
torch.save(eta, '../data/prior_eta.pt')
torch.save(omega_tt, '../data/prior_omega_tt.pt')
torch.save(omega_next_t, '../data/prior_omega_next_t.pt')
torch.save(utils.block2flat(joint_mean.unsqueeze(2).unsqueeze(-1)), '../data/prior_mean.pt')
torch.save(utils.block2flat(joint_cov), '../data/prior_cov.pt')

mean, eta, omega_tt, omega_next_t, eta_new, omega_tt_new = ks.smoother(linear_ds.observations)
torch.save(eta_new, '../data/smoothed_eta.pt')
torch.save(omega_tt_new, '../data/smoothed_omega_tt.pt')
torch.save(omega_next_t, '../data/smoothed_omega_next_t.pt')

joint_omega_smoothed = ks.joint_omega(omega_tt_new[0], omega_next_t[0])
joint_cov_smoothed = torch.inverse(joint_omega_smoothed)


print(f'joint smoother: {joint_cov_smoothed[t*ks.state_dim:(t+1)*ks.state_dim, (t+1)*ks.state_dim:(t+2)*ks.state_dim]}')
print(f'joint smoother: {joint_cov_smoothed[t*ks.state_dim:(t+1)*ks.state_dim, (t)*ks.state_dim:(t+1)*ks.state_dim]}')


#
# ks.F = torch.eye(ks.state_dim, ks.state_dim).unsqueeze(0).repeat(ks.batch_dim, 1, 1)
# ks.EM(linear_ds.observations, 20, update='F')


dl = DataLoader(linear_ds, batch_size=2)
model = KSMLE(ks.state_dim, lr=1e-3)

trainer = pl.Trainer(max_epochs=10)

trainer.fit(model, dl)

