import wandb
import argparse

from train import train_node_classifier
from hyperopt import optimalization
from datasets.load_datasets import get_gtex_disgenet_dataset, get_gtex_disgenet_test_dataset
from config import Config

import pytorch_lightning as pl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, choices=["basic"], default="basic")
    parser.add_argument('-epoch', type=int)
    parser.add_argument('--opt', action='store_true', help="If given, then optimalization will run.")
    parser.add_argument('-opt-step', type=int)
    #parser.add_argument('--val-size', type=float, help="The validation dataset size in percentage. Like 0.2: then 20 percentage of dataset will be validation.", default=0.1)
    parser.add_argument('--test-dataset', action='store_true', help="If user want to use the test dataset (wich is a small dataset for testing code).")
    parser.add_argument('--process-dataset', action='store_true', help="If given dataset will be recreated.")
    args = parser.parse_args()


    if args.process_dataset:
        Config.process_files = True

    
    if args.model:
        Config.model_name = args.model

    if args.epoch:
        Config.epochs = args.epoch
    
    if args.opt_step:
        Config.optimalization_step = args.opt_step

    diseases = ["C0006142", "C0036341"]

    wandb.login(key=Config.wandb_api_key)

    for disease in diseases:
        folder_path = "data/processed"
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):  # Ensure it's a file
                os.remove(file_path)

        if args.test_dataset:
            dataset = get_gtex_disgenet_test_dataset(disease)
            Config.test_dataset = True
        else:
            dataset = get_gtex_disgenet_dataset(disease)

        Config.test_dataset = False

        Config.in_channels = dataset.num_node_features
        Config.out_channels = dataset[0].y.shape[1]
        
        if args.opt:
            optimalization(dataset=dataset)
        else:
            node_mlp_model, node_mlp_result = train_node_classifier(dataset=dataset)

    wandb.finish()
