import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import numpy as np
import os

from structuredKS import utils

class LGSSM:

    def __init__(self, initial_mean, initial_precision, transition_model, transition_precision,
                 observation_model, observation_precision):

        # prior distribution for t=0
        self.p0 = MultivariateNormal(initial_mean, precision_matrix=initial_precision)

        self.F = transition_model
        self.Q_inv = transition_precision
        self.H = observation_model
        self.R_inv = observation_precision

        self.sample_intial()

    def sample_intial(self):
        self.state = self.p0.rsample()
        return self.state

    def sample_transition(self):
        p = MultivariateNormal(self.F @ self.state, precision_matrix=self.Q_inv)
        self.state = p.rsample()
        return self.state

    def sample_observation(self):
        p = MultivariateNormal(self.H @ self.state, precision_matrix=self.R_inv)
        return p.rsample()

    def generate_sequence(self, T):
        states = torch.zeros(T, self.H.shape[1])
        observations = torch.zeros(T, self.H.shape[0])

        states[0] = self.sample_intial()
        observations[0] = self.sample_observation()

        for t in range(1, T):
            states[t] = self.sample_transition()
            observations[t] = self.sample_observation()

        return states, observations



class LinearTransport(Dataset):

    def __init__(self, config, task='train', transform=None, load_data_from=''):

        # set random seed
        seed = config['seed'] + 1 if task == 'test' else config['seed']
        torch.manual_seed(seed)

        # general settings
        self.n_data = config[f'n_{task}']
        self.T = config['T']
        self.grid_size = config['grid_size']
        self.N = self.grid_size * self.grid_size
        self.M = config.get('M', self.N)

        # grid for field generation
        self.x = np.arange(self.grid_size)
        self.y = np.arange(self.grid_size)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.coords = torch.from_numpy(np.stack([self.xx, self.yy], axis=-1)).float()

        self.xspacing = self.x[1] - self.x[0]
        self.yspacing = self.y[1] - self.y[0]

        self.dt = config.get('dt', 0.1)

        self.initial_mean = torch.zeros((self.n_data, self.N), dtype=torch.float32)
        self.initial_precision = torch.zeros((self.n_data, self.N, self.N), dtype=torch.float32)
        self.transition_model = torch.zeros((self.n_data, self.N, self.N), dtype=torch.float32)
        self.transition_precision = torch.zeros((self.n_data, self.N, self.N), dtype=torch.float32)
        self.observation_model = torch.zeros((self.n_data, self.M, self.N), dtype=torch.float32)
        self.observation_precision = torch.zeros((self.n_data, self.M, self.M), dtype=torch.float32)
        self.v = torch.zeros((self.n_data, 2), dtype=torch.float32)

        self.latent_states = torch.zeros((self.n_data, self.T, self.N), dtype=torch.float32)
        self.observations = torch.zeros((self.n_data, self.T, self.M), dtype=torch.float32)

        self.initial_modes = config.get('initial_modes', 3)


        if os.path.isdir(load_data_from):
            self.load(load_data_from)
        else:
            # generate data
            vx, vy = torch.rand(2) - 0.5 # same velocity field for all sequences
            for i in range(self.n_data):
                print(f'generate sequence {i}')

                # setup LGSSM
                mean = self.generate_mean() * config.get('initial_scale', 1)
                self.initial_mean[i] = mean.view(-1)
                kernel = torch.tensor([[0, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 0]]).view(1, 1, 3, 3)
                L = utils.conv2matrix(kernel, mean.unsqueeze(0).shape) / config.get('initial_noise_level', 1e-5)
                self.initial_precision[i] = L.T @ L

                self.transition_model[i], _, self.v[i] = self.generate_F(vx, vy)
                self.transition_precision[i] = torch.eye(self.N) / config.get('transition_noise_level', 1e-5)

                self.observation_precision[i] = torch.eye(self.M) / config.get('observation_noise_level', 1e-5)
                idx = np.random.choice(range(self.N), self.M, replace=False)
                self.observation_model[i] = torch.eye(self.N)[idx]

                lgssm = LGSSM(self.initial_mean[i], self.initial_precision[i],
                              self.transition_model[i], self.transition_precision[i],
                              self.observation_model[i], self.observation_precision[i])

                # sample latent states and observations
                self.latent_states[i], self.observations[i] = lgssm.generate_sequence(self.T)

                # generate latent states
                #self.latent_states[i], self.transitions[i] = self.generate_fields()

            # generate observations
            #self.apply_observation_model()


    # def apply_observation_model(self):
    #
    #     observations = self.observation_model.unsqueeze(1).repeat(1, self.T, 1, 1) @ \
    #                         self.latent_states.unsqueeze(-1)
    #     noise = torch.randn(self.observations.size()) * self.observation_noise.unsqueeze(1)
    #     self.observations = observations + noise


    def generate_F(self, vx, vy):
        # vx, vy = torch.rand(2) - 0.5
        v = torch.tensor([vx, vy])

        cx = self.dt * 2 * self.xspacing
        cy = self.dt * 2 * self.yspacing

        conv_kernel = torch.tensor([[0, cy * vy, 0],
                                    [cx * vx, 1, -cx * vx],
                                    [0, -cy * vy, 0]])

        conv_kernel = conv_kernel.view(1, 1, 3, 3) # apply with F.conv2d(padded_img, kernel)

        F_matrix = utils.conv2matrix(conv_kernel, (1, self.grid_size, self.grid_size))  # apply with F @ flattened_img

        return F_matrix, conv_kernel, v

    def generate_mean(self):

        modes = 0.5 * self.grid_size * torch.rand((self.initial_modes, 2)) + 0.25 * self.grid_size
        sigmas = torch.rand(self.initial_modes) + 0.1 * self.grid_size

        potentials = torch.zeros(self.xx.shape)
        for m, s in zip(modes, sigmas):
            dst = self.coords - m
            potentials += torch.exp(-((dst * dst).sum(-1) / (s**2))) * (torch.rand(1) + 0.5)

        return potentials


    # def generate_fields(self):
    #     # transport scalar quantity along vector field
    #
    #     # setup scalar quantity
    #     states = torch.zeros(self.T, self.grid_size, self.grid_size)
    #     states[0] = self.generate_mean(loc, scale)
    #
    #     F_matrix, conv_kernel = self.generate_F()
    #     pads = torch.ones(4)
    #
    #     for t in range(1, self.T):
    #         # update state
    #         padded_input = F.pad(torch.ones(states[t-1]), pads)
    #         states[t] = F.conv2d(padded_input, conv_kernel)
    #
    #     # flatten to 1D states
    #     states = states.flatten(start_dim=1)
    #     return states, F_matrix



    def save(self, dir):

        os.makedirs(dir, exist_ok=True)

        torch.save(self.initial_mean, os.path.join(dir, 'initial_mean.pt'))
        torch.save(self.initial_precision, os.path.join(dir, 'initial_precision.pt'))
        torch.save(self.transition_model, os.path.join(dir, 'transition_model.pt'))
        torch.save(self.transition_precision, os.path.join(dir, 'transition_precision.pt'))
        torch.save(self.latent_states, os.path.join(dir, 'latent_states.pt'))
        torch.save(self.observations, os.path.join(dir, 'observations.pt'))
        torch.save(self.observation_model, os.path.join(dir, 'observation_model.pt'))
        torch.save(self.observation_precision, os.path.join(dir, 'observation_precision.pt'))
        torch.save(self.v, os.path.join(dir, 'v.pt'))

    def load(self, dir):

        # load existing data
        initial_mean = torch.load(os.path.join(dir, 'initial_mean.pt'))
        initial_precision = torch.load(os.path.join(dir, 'initial_precision.pt'))
        transition_model = torch.load(os.path.join(dir, 'transition_model.pt'))
        transition_precision = torch.load(os.path.join(dir, 'transition_precision.pt'))
        latent_states = torch.load(os.path.join(dir, 'latent_states.pt'))
        observations = torch.load(os.path.join(dir, 'observations.pt'))
        observation_model = torch.load(os.path.join(dir, 'observation_model.pt'))
        observation_precision = torch.load(os.path.join(dir, 'observation_precision.pt'))
        v = torch.load(os.path.join(dir, 'v.pt'))

        assert initial_mean.shape[0] >= self.n_data
        assert initial_precision.shape[0] >= self.n_data
        assert transition_model.shape[0] >= self.n_data
        assert transition_precision.shape[0] >= self.n_data
        assert observation_model.shape[0] >= self.n_data
        assert observation_precision.shape[0] >= self.n_data
        assert torch.all(torch.Tensor(list(latent_states.shape[:2])) >= torch.Tensor([self.n_data, self.T]))
        assert torch.all(torch.Tensor(list(observations.shape[:2])) >= torch.Tensor([self.n_data, self.T]))
        assert v.shape[0] >= self.n_data

        self.initial_mean = initial_mean[:self.n_data]
        self.initial_precision = initial_precision[:self.n_data]
        self.transition_model = transition_model[:self.n_data]
        self.transition_precision = transition_precision[:self.n_data]
        self.observation_model = observation_model[:self.n_data]
        self.observation_precision = observation_precision[:self.n_data]
        self.latent_states = latent_states[:self.n_data, :self.T]
        self.observations = observations[:self.n_data, :self.T]
        self.v = v[:self.n_data]



    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'initial_mean': self.initial_mean[idx],
            'initial_precision': self.initial_precision[idx],
            'transition_model': self.transition_model[idx],
            'transition_precision': self.transition_precision[idx],
            'latent_states': self.latent_states[idx],
            'observations': self.observations[idx],
            'observation_model': self.observation_model[idx],
            'observation_precision': self.observation_precision[idx],
            'idx': idx
        }

        return sample