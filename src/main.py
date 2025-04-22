from mappers.idmapper import IdMapper
import wandb
import argparse

from train import train_node_classifier
from hyperopt import optimalization
from datasets.load_datasets import get_gtex_disgenet_dataset, get_gtex_disgenet_test_dataset
from config import Config, ModelTypes

import pytorch_lightning as pl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, choices=[ModelTypes.BASIC, ModelTypes.CLS_WEIGHT], default=ModelTypes.CLS_WEIGHT)
    parser.add_argument('-disease', type=str)
    parser.add_argument('-epoch', type=int)
    parser.add_argument('--opt', action='store_true', help="If given, then optimalization will run.")     
    parser.add_argument('-opt-step', type=int)
    #parser.add_argument('--val-size', type=float, help="The validation dataset size in percentage. Like 0.2: then 20 percentage of dataset will be validation.", default=0.1)
    parser.add_argument('--test-dataset', action='store_true', help="If user want to use the test dataset (wich is a small dataset for testing code).")
    parser.add_argument('--new-dataset', action='store_true', help="If given, then the graph dataset will be recrated.")

    args = parser.parse_args()
    
    Config.model_name = args.model

    if args.epoch:
        Config.epochs = args.epoch
    
    if args.opt_step:
        Config.optimalization_step = args.opt_step

    if args.new_dataset:
        Config.process_files = True

    if args.test_dataset:
        dataset = get_gtex_disgenet_test_dataset()
        Config.test_dataset = True
    else:
        dataset = get_gtex_disgenet_dataset()

    if args.disease:
        disease_id = dataset.mapper.diseases_id_to_idx_map()[args.disease]
        Config.disease_idx = disease_id
        Config.pos_class_weight = dataset[0].train_mask[:,disease_id].sum() / dataset[0].y[:,disease_id].sum()
    print(disease_id, Config.pos_class_weight)

    Config.test_dataset = False
    Config.in_channels = dataset.num_node_features
    Config.out_channels = dataset[0].y.shape[1]
    
    wandb.login(key=Config.wandb_api_key)

    if args.opt:
        optimalization(dataset=dataset)
    else:
        train_node_classifier(dataset=dataset)

    wandb.finish()
