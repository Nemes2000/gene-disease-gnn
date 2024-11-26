import wandb
import argparse

from train import train_node_classifier
from hyperopt import optimalization
from datasets.load_datasets import get_gtex_disgenet_dataset, get_gtex_disgenet_test_dataset
from config import Config

import pytorch_lightning as pl

if __name__ == "__main__":
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, choices=["basic"])
    parser.add_argument('-epoch', type=int)
    parser.add_argument('--opt', action='store_true', help="If given optimalization will run.")
    parser.add_argument('--test-dataset', action='store_true', help="If given test dataset will be used.")
    args = parser.parse_args()

    wandb.login(key=Config.wandb_api_key)

    print(args)

    if args.test_dataset:
        dataset = get_gtex_disgenet_test_dataset()
    else:
        dataset = get_gtex_disgenet_dataset()

    Config.in_channels = dataset.num_node_features
    Config.out_channels = dataset[0].y.shape[1]
    
    if args.model:
        Config.model_name = args.model

    if args.epoch:
        Config.epochs = args.epoch

    if args.opt:
        optimalization(dataset=dataset)
    else:
        node_mlp_model, node_mlp_result = train_node_classifier(dataset=dataset)

    wandb.finish()
