import torch
import torch_geometric as ptg
import numpy as np
import json
import os
import os.path as osp
import time
import argparse
import wandb
import copy
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
import pickle

from stdgmrf.models.dgmrf import *
import constants_dgmrf as constants
from stdgmrf import utils
from stdgmrf.datasets.dummy_dataset import DummyDataset
from callbacks import *


def print_params(model, config, header=None):
    if header:
        print(header)

    print("Aggregation weights:")
    for layer_i, layer in enumerate(model.layers):
        print("Layer {}".format(layer_i))
        if hasattr(layer, "activation_weight"):
            print("non-linear weight: {:.4}".format(layer.activation_weight.item()))
        else:
            print("self: {:.4}, neighbor: {:.4}".format(
                layer.self_weight[0].item(), layer.neighbor_weight[0].item()))

        if hasattr(layer, "degree_power"):
            print("degree power: {:.4}".format(layer.degree_power[0].item()))


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="conf", config_name="config")
def run_dgmrf(config: DictConfig):

    print(f'hydra working dir: {os.getcwd()}')

    if 'wandb_run' in config:
        # load config of pre-trained model
        model_path, full_config = utils.get_wandb_model(config['wandb_run'], return_config=True)

        wandb_config = eval(full_config['config'].replace("'", '"'))
        
        # update wandb config with overrides
        overrides = HydraConfig.get().overrides.task
        print(overrides)
        
        parser = OverridesParser.create()
        parsed_overrides = parser.parse_overrides(overrides=overrides)

        for override in parsed_overrides:
            key = override.key_or_group
            value = override.value()
            wandb_config[key] = value
        
        config = wandb_config
        config = OmegaConf.create(config)

    print(config)

    utils.seed_all(config['seed'])

    if not config['device'] == "cpu" and torch.cuda.is_available():
        print('Use GPU')
        # Make all tensors created go to GPU

        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        # For reproducability on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = 'cuda'
    else:
        print('Use CPU.')
        device = 'cpu'

    print('setup wandb')

    # Init wandb
    wandb_name = f"{config['transition_type']}-" \
                 f"indepT={config['independent_time']}-layers={config['n_layers']}-bias={config['use_bias']}-" \
                 f"{config['n_transitions']}trans-{config['dataset']}-{time.strftime('%H-%M')}"
    run = wandb.init(project=config['experiment'], config=config, name=wandb_name)

    print('load data')

    dataset_dict = utils.load_dataset(config["dataset"], config["data_dir"], device=device)

    # initialize model based on given config
    model = setup_model(config, device)

    wandb_logger = WandbLogger()

    if ("true_posterior_mean" in dataset_dict) and ("gt" in dataset_dict):

        true_mean = dataset_dict["true_posterior_mean"].squeeze()
        test_mask = torch.logical_not(joint_mask)

        gt = dataset_dict["gt"]
        residuals = (gt - true_mean)

        wandb.run.summary["test_mae_optimal"] = residuals[test_mask].abs().mean().item()
        wandb.run.summary["test_rmse_optimal"] = torch.pow(residuals[test_mask], 2).mean().sqrt().item()
        wandb.run.summary["test_mape_optimal"] = (residuals[test_mask] / gt[test_mask]).abs().mean().item()

    callbacks = []

    trainer = pl.Trainer(
        max_epochs=int(config["n_iterations"] / config["val_interval"]),
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,
        accelerator='gpu' if device == 'cude' else 'cpu',
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=config.get('gradient_clip_val', 0.0)
    )
    
    for param_name, param_value in model.vi_dist.dynamics.state_dict().items():
        print(f'{param_name}: {param_value}')

    if 'wandb_run' in config:
        # load pre-trained model
        model_path = utils.get_wandb_model(config.get('wandb_run'))
        model.load_state_dict(torch.load(model_path))
    else:
        # train model
        ds_train = DummyDataset(dataset_dict['train_masks'].reshape(-1), config["val_interval"])
        dl_train = DataLoader(ds_train, batch_size=1, shuffle=False)
        ds_val = DummyDataset(dataset_dict['val_masks'].reshape(-1), 1)
        dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)

        start = timer()
        trainer.fit(model, dl_train, dl_val)
        end = timer()

        wandb.run.summary["train_iter_time"] = (end - start) / config["n_iterations"]

    if config.get('final', False):
        ds_test = DummyDataset(dataset_dict['test_masks'].reshape(-1), 1)
        dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

        if config.get('save_prediction', False): 
            results = trainer.predict(model, dl_test, return_predictions=True)
        else:
            print('evaluation on test set')
            start = timer()
            trainer.test(model, dl_test)
            end = timer()

            wandb.run.summary["inference_time"] = (end - start)
    else:
        ds_val = DummyDataset(dataset_dict['val_masks'].reshape(-1), 1)
        dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)

        if config.get('save_prediction', False):
            results = trainer.predict(model, dl_val, return_predictions=True)
        else:
            trainer.test(model, dl_val)
    
    for param_name, param_value in model.dgmrf.state_dict().items():
        print("{}: {}".format(param_name, param_value))

    print(f'noise var = {model.noise_var}')

    ckpt_dir = osp.join(config['output_dir'], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # save results
    if config.get('save_prediction', False):
        result_path = osp.join(ckpt_dir, config['experiment'], run.id, 'results')
        os.makedirs(result_path, exist_ok=True)
        with open(osp.join(result_path, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save as artifact for version control
        artifact = wandb.Artifact(f'results-{run.id}', type='results')
        artifact.add_dir(result_path)
        run.log_artifact(artifact)

    # save model
    model_path = osp.join(ckpt_dir, config['experiment'], run.id, 'models')
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))

    # save as artifact for version control
    artifact = wandb.Artifact(f'model-{run.id}', type='models')
    artifact.add_dir(model_path)
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    run_dgmrf()
