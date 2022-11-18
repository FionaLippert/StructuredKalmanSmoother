import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os

from structuredKS import utils

class LinearTransport(Dataset):

    def __init__(self, config, task='train', transform=None, load_data_from='', potential_modes=10):

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

        self.transitions = torch.zeros((self.n_data, self.N, self.N), dtype=torch.float32)
        self.latent_states = torch.zeros((self.n_data, self.T, self.N), dtype=torch.float32)
        self.observation_model = torch.zeros((self.n_data, self.M, self.N), dtype=torch.float32)
        self.observation_noise = torch.zeros((self.n_data, self.M), dtype=torch.float32)


        if os.path.isdir(load_data_from):
            self.load(load_data_from)
        else:
            # generate data
            for i in range(self.n_data):
                print(f'generate sequence {i}')

                # define observation model
                self.observation_cov[i] = 0.1 * torch.eye(self.M)
                idx = np.random.choice(range(self.N), self.M)
                self.observation_model[i] = torch.eye(self.M)[idx]

                # generate latent states
                self.latent_states[i], self.transitions[i] = self.generate_fields(potential_modes)

            # generate observations
            self.apply_observation_model()


    def apply_observation_model(self):

        observations = self.observation_model.unsqueeze(1).repeat(1, self.T, 1, 1) @ \
                            self.latent_states.unsqueeze(-1)
        noise = torch.randn(self.observations.size()) * self.observation_noise.unsqueeze(1)
        self.observations = observations + noise


    def generate_F(self):
        vx, vy = torch.rand(2) - 0.5

        cx = self.dt * 2 * self.xspacing
        cy = self.dt * 2 * self.yspacing

        conv_kernel = torch.tensor([[0, cy * vy, 0],
                                    [cx, 1, -cx * vx],
                                    [0, -cy * vy, 0]])

        conv_kernel = conv_kernel.view(1, 1, 3, 3) # apply with F.conv2d(padded_img, kernel)

        F_matrix = utils.conv2matrix(conv_kernel, (1, self.grid_size, self.grid_size))  # apply with F @ flattened_img

        return F_matrix, conv_kernel


    def generate_fields(self, n_modes):
        # transport scalar quantity along vector field

        # setup scalar quantity
        modes = 6 * (torch.rand((n_modes, 2)) - 0.5)
        sigmas = torch.rand(n_modes) * 1 + 0.5
        states = torch.zeros(self.T, self.grid_size, self.grid_size)
        states[0] = self.generate_potential(modes, sigmas)

        F_matrix, conv_kernel = self.generate_F()
        pads = torch.ones(4)

        for t in range(1, self.T):
            # update state
            padded_input = F.pad(torch.ones(states[t-1]), pads)
            states[t] = F.conv2d(padded_input, conv_kernel)

        # flatten to 1D states
        states = states.flatten(start_dim=1)
        return states, F_matrix



    def save(self, dir):

        os.makedirs(dir, exist_ok=True)

        torch.save(self.transitions, os.path.join(dir, 'transitions.pt'))
        torch.save(self.latent_states, os.path.join(dir, 'latent_states.pt'))
        torch.save(self.observations, os.path.join(dir, 'observations.pt'))
        torch.save(self.observation_model, os.path.join(dir, 'observation_model.pt'))
        torch.save(self.observation_noise, os.path.join(dir, 'observation_noise.pt'))

    def load(self, dir):

        # load existing data
        transitions = torch.load(os.path.join(dir, 'transitions.pt'))
        latent_states = torch.load(os.path.join(dir, 'latent_states.pt'))
        observations = torch.load(os.path.join(dir, 'observations.pt'))
        observation_model = torch.load(os.path.join(dir, 'observation_model.pt'))
        observation_noise = torch.load(os.path.join(dir, 'observation_noise.pt'))

        assert transitions.shape[0] >= self.n_data
        assert observation_model.shape[0] >= self.n_data
        assert observation_noise.shape[0] >= self.n_data
        assert torch.all(torch.Tensor(list(latent_states.shape[:2])) >= torch.Tensor([self.n_data, self.T]))
        assert torch.all(torch.Tensor(list(observations.shape[:2])) >= torch.Tensor([self.n_data, self.T]))

        self.transitions = transitions[:self.n_data]
        self.observation_model = observation_model[:self.n_data]
        self.observation_noise = observation_noise[:self.n_data]
        self.latent_states = latent_states[:self.n_data, :self.T]
        self.observations = observations[:self.n_data, :self.T]



    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'transition': self.transitions[idx],
            'latent_states': self.latent_states[idx],
            'observations': self.observations[idx],
            'observation_model': self.observation_model[idx],
            'observation_noise': self.observation_noise[idx]
        }

        return sample