from torch_geometric.transforms import NormalizeFeatures

from config import Config
from datasets.gene_dataset import GeneDataset 

def get_gtex_disgenet_dataset():
    dataset = GeneDataset(
        root="./data",
        filenames=["gtex_genes.csv", "gene_graph.csv", "disgenet_with_gene_id.csv"],
        test_size=Config.train_test_split,
        val_size=Config.test_val_split,
        process_files=Config.process_files,
        transform=NormalizeFeatures())
    return dataset

def get_gtex_disgenet_test_dataset():
    dataset = GeneDataset(
        root="./data",
        filenames=["gtex_genes_test.csv", "gene_graph_test.csv", "disgenet_test.csv"],
        test_size=Config.train_test_split,
        val_size=Config.test_val_split,
        process_files=Config.process_files,
        transform=NormalizeFeatures())
    return dataset