import numpy as np

DS_DIR = "../datasets"
WANDB_PROJECT = "spatiotemporal_dgmrf_experiments"

RAW_DATA_DIR = "../raw_data"

# Custom views that should be plotted for datasets
DATASET_ZOOMS = {
        "cal": np.array([
            [[-0.6707,-0.0590], [-0.4781, 0.2289]],
            [[-0.5235, -0.1929], [0.0265, 0.2339]],
            [[0.1009, -0.9591], [0.4370, -0.7207]],
        ])
}