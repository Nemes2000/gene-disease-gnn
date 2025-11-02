import pandas as pd
import wandb
import argparse

from models.lightning_gnn_model import LightningGNNModel
from train import test_node_classifier, train_node_classifier
from hyperopt import optimalization
from datasets.load_datasets import get_gtex_disgenet_dataset, get_gtex_disgenet_test_dataset
from config import Config, ModelTypes

import pytorch_lightning as pl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, choices=[ModelTypes.BASIC.value, ModelTypes.CLS_WEIGHT.value, ModelTypes.MULTITASK.value], default=ModelTypes.MULTITASK.value)
    parser.add_argument('-gnn_layer', type=str, choices=["GCN", "GraphSAGE"], default="GraphSAGE")
    parser.add_argument('-model_ckpt_name', type=str)
    parser.add_argument('-disease', type=str)
    parser.add_argument('-pr_disease', type=str)
    parser.add_argument('-aux_diseases', nargs="+", type=str)
    parser.add_argument('--all_diseases', action="store_true", help="If given, then all diseases will be used as auxiliary tasks except the primary disease.")
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

    if args.gnn_layer:
        Config.gnn_layer_type = args.gnn_layer

    if args.model == ModelTypes.MULTITASK and args.pr_disease:
        Config.out_channels = 43
        Config.pr_disease_idx = dataset.mapper.diseases_id_to_idx_map()[args.pr_disease]
        Config.wandb_project_name += "_" + str(Config.pr_disease_idx)

        all_g = dataset[0].y.shape[0]
        Config.pr_pos_class_weight =(all_g - dataset[0].y[:,Config.pr_disease_idx].sum())/dataset[0].y[:, Config.pr_disease_idx].sum()
        
        if args.aux_diseases:
            Config.aux_disease_idxs =  [dataset.mapper.diseases_id_to_idx_map()[aux_disease] for aux_disease in args.aux_diseases]
            Config.aux_task_num = len(args.aux_diseases)
            Config.aux_pos_class_weights = [(all_g - dataset[0].y[:,idx].sum())/dataset[0].y[:, idx].sum() for idx in Config.aux_disease_idxs]
        if args.all_diseases:
            Config.aux_disease_idxs = [i for i in range(dataset[0].y.shape[1]) if i != Config.pr_disease_idx]
            Config.aux_task_num = all_g - 1
            Config.aux_pos_class_weights = [(all_g - dataset[0].y[:,idx].sum())/dataset[0].y[:, idx].sum() for idx in Config.aux_disease_idxs]

    if args.disease:
        disease_idx = dataset.mapper.diseases_id_to_idx_map()[args.disease]
        Config.disease_idx = disease_idx
        Config.pos_class_weight = (dataset[0].y.shape[1] - dataset[0].y[:,disease_idx].sum())/dataset[0].y[:, disease_idx].sum()
        print(disease_idx, Config.pos_class_weight)
        print("sum pos: ",dataset[0].train_mask[:,disease_idx].sum())
        print("y shape: ",dataset[0].y.shape)

    
    if args.model == ModelTypes.BASIC or args.model == ModelTypes.CLS_WEIGHT:
        Config.wandb_project_name = "basic_gnn"
        Config.out_channels = dataset[0].y.shape[1]

    Config.test_dataset = False
    Config.in_channels = dataset.num_node_features

    # df = pd.DataFrame(dataset[0].y.numpy())
    # df.to_csv("results/y_matrix.csv", index=False)

    if args.model_ckpt_name:
        test_node_classifier(dataset=dataset, model_ckpt_name=args.model_ckpt_name)
    else:
        wandb.login(key=Config.wandb_api_key)

        if args.opt:
            optimalization(dataset=dataset)
        else:
            train_node_classifier(dataset=dataset)

        wandb.finish()