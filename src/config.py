import torch_geometric.nn as geom_nn

import torch
import os

class Config():
    min_disease_s_gene_number = 7
    train_test_split = 0.5
    test_val_split = 0.5

    learning_rate = 0.01
    weight_decay = 5e-4
    num_classes = 2

    epochs = 1
    
    avail_gpus = min(1, torch.cuda.device_count())

    checkpoint_path = os.environ.get("PATH_CHECKPOINT", "../../data/saved_models/")
    raw_data_path = os.environ.get("PATH_CHECKPOINT", "../../data/raw/")

    gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

    wandb_api_key = "e1f878235d3945d4141f9f8e5af41d712fca6eba"
