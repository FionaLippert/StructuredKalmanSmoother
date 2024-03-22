from stdgmrf.datasets import linear
from stdgmrf.models.KS import KalmanSmoother, JointKalmanSmoother, KSMLE
from stdgmrf import utils
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
    'transition_noise_level': 0.01,
    'observation_noise_level': 0.01,
    'initial_noise_level': 0.1,
    'initial_scale': 10
}

linear_ds = linear.LinearTransport(config)
linear_ds.save('../data')

# test Kalman smoother
ks = KalmanSmoother(linear_ds.initial_mean, torch.cholesky_inverse(torch.linalg.cholesky(linear_ds.initial_precision)),
                    linear_ds.transition_model, torch.cholesky_inverse(torch.linalg.cholesky(linear_ds.transition_precision)),
                    linear_ds.observation_model, torch.cholesky_inverse(torch.linalg.cholesky(linear_ds.observation_precision)))

mean, cov = ks.prior(config['T'])
torch.save(mean, '../data/prior_mean_marginal.pt')
torch.save(cov, '../data/prior_cov_marginal.pt')

mean, cov, cov_lagged = ks.smoother(linear_ds.observations)
torch.save(mean, '../data/smoothed_mean.pt')
torch.save(cov, '../data/smoothed_cov.pt')



A = linear_ds.transition_model[0]
weights = A.to_sparse().coalesce().values()

bs = 1
dl = DataLoader(linear_ds, batch_size=bs)
model = KSMLE(ks.state_dim, weights, use_gnn=False, temporal_graph=A, lr=1e-3)

ks = JointKalmanSmoother(linear_ds.initial_mean, linear_ds.initial_precision,
                                 model.transition, linear_ds.transition_precision,
                                 linear_ds.observation_model, linear_ds.observation_precision, linear_ds.N)

joint_mean, joint_eta, joint_omega, updated_eta, updated_omega, joint_R_inv, joint_H = ks.smoother(linear_ds.observations)
joint_cov = torch.cholesky_inverse(torch.linalg.cholesky(joint_omega))

updated_cov = torch.cholesky_inverse(torch.linalg.cholesky(updated_omega))
updated_mean = (updated_cov @ updated_eta.unsqueeze(-1))

cov_y = joint_H @ joint_cov @ joint_H.transpose(1, 2) + torch.cholesky_inverse(torch.linalg.cholesky(joint_R_inv))
mean_y = (joint_H @ joint_mean.unsqueeze(-1)).squeeze(-1)

p = torch.distributions.multivariate_normal.MultivariateNormal(mean_y, cov_y)
nll = -(p.log_prob(linear_ds.observations.reshape(linear_ds.n_data, -1))).mean() / (linear_ds.N)
print(f'optimal nll = {nll}')

torch.save(joint_mean, '../data/prior_joint_mean.pt')
torch.save((joint_cov @ joint_eta.unsqueeze(-1)).squeeze(-1), '../data/prior_joint_mean_from_eta.pt')
torch.save(joint_cov, '../data/prior_joint_cov.pt')
torch.save(updated_mean, '../data/smoothed_joint_mean.pt')
torch.save(updated_cov, '../data/smoothed_joint_cov.pt')


trainer = pl.Trainer(max_epochs=50)

model.transition.reset_weights()

ks = JointKalmanSmoother(linear_ds.initial_mean, linear_ds.initial_precision,
                                 model.transition, linear_ds.transition_precision,
                                 linear_ds.observation_model, linear_ds.observation_precision, linear_ds.N)

joint_mean, joint_eta, joint_omega, updated_eta, updated_omega, joint_R_inv, joint_H = ks.smoother(linear_ds.observations)
updated_cov = torch.cholesky_inverse(torch.linalg.cholesky(updated_omega))
updated_mean = (updated_cov @ updated_eta.unsqueeze(-1)).squeeze(-1)

torch.save(updated_mean, '../data/smoothed_joint_mean_before_MLE.pt')
torch.save(updated_cov, '../data/smoothed_joint_cov_before_MLE.pt')

trainer.fit(model, dl)

ks = JointKalmanSmoother(linear_ds.initial_mean, linear_ds.initial_precision,
                                 model.transition, linear_ds.transition_precision,
                                 linear_ds.observation_model, linear_ds.observation_precision, linear_ds.N)

joint_mean, joint_eta, joint_omega, updated_eta, updated_omega, joint_R_inv, joint_H = ks.smoother(linear_ds.observations)
updated_cov = torch.cholesky_inverse(torch.linalg.cholesky(updated_omega))
updated_mean = (updated_cov @ updated_eta.unsqueeze(-1)).squeeze(-1)

torch.save(updated_mean, '../data/smoothed_joint_mean_MLE.pt')
torch.save(updated_cov, '../data/smoothed_joint_cov_MLE.pt')


