from torch_geometric.transforms import NormalizeFeatures

import wandb

from train import train_node_classifier
from datasets.gene_dataset import GeneDataset
from config import Config

if __name__ == "__main__":

    wandb.login(key=Config.wandb_api_key)

    #if args.valami == '...':  --> argumentumok megadása lehet
    dataset = GeneDataset(
        root="./data",
        filenames=["gtex_genes.csv", "gene_graph.csv", "disgenet_with_gene_id.csv"],
        test_size=0.2,
        val_size=0.0,
        transform=NormalizeFeatures())

    node_mlp_model, node_mlp_result = train_node_classifier(
        model_name="GCN", dataset=dataset, c_hidden=16, num_layers=2, dp_rate=0.1
    )
