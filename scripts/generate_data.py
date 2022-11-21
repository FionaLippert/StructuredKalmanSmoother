from structuredKS.datasets import linear

config = {
    'seed': 1234,
    'n_train': 5,
    'grid_size': 25,
    'M': 200,
    'T': 10,
    'dt': 0.2,
    'initial_modes': 5,
    'transition_noise_level': 0.0001,
    'observation_noise_level': 0.0001,
    'initial_noise_level': 0.0001,
    'initial_scale': 1

}

linear_ds = linear.LinearTransport(config)
linear_ds.save('../data')
