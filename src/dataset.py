import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from sklearn.model_selection import train_test_split

class GeneDataset(Dataset):
    def __init__(self, root, filenames, test_size, val_size, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.test_size = test_size
        self.val_size = val_size
        self.filenames = filenames
        super(GeneDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filenames

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.test:
            return [F'{file_name}_test' for file_name in self.raw_paths]
        else:
            return self.raw_paths

    def download(self):
        pass

    def process(self):
        self.genes = pd.read_csv(self.raw_paths[0], sep="\t")
        self.genes = self.genes.drop(columns="Description")
        self.edges = pd.read_csv(self.raw_paths[1], sep="\t")

        node_feats = self._get_node_features(self.genes)
        edge_feats = self._get_edge_features(self.edges)
        edge_index = self._get_adjacency_info(self.edges)

        y = self._get_test_labels(self.genes)
        train_mask, val_mask, test_mask = self._get_train_val_test_mask(self.genes) 

        data = Data(x=node_feats, 
                    edge_index=edge_index,
                    edge_attr=edge_feats,
                    test_mask=test_mask, val_mask=val_mask, train_mask=train_mask, y=y)
         
        if self.test:
            torch.save(data, os.path.join(self.processed_dir, 'graph_test.pt'))
        else:
            torch.save(data, os.path.join(self.processed_dir, 'graph.pt'))


    def _get_train_val_test_mask(self, genes):
        X_train, X_test, y_train, y_test = train_test_split(, , test_size=self.test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=42)
        
        return np.ones()
    

    def _get_test_labels(self, genes):
        return

    def _get_node_features(self, genes):
        genes["genes"] = genes["genes"].str[4:].astype(int)
        all_node_feats = genes.values.tolist()
        all_node_feats = np.asarray(all_node_feats)
        
        return torch.tensor(all_node_feats, dtype=torch.int64)

    def _get_edge_features(self, edges):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = edges["combined_score"].tolist()
        return torch.tensor(all_edge_feats, dtype=torch.float)


    def _get_adjacency_info(self, edges):
        """
        We want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        gene_1 = edges["gene1"].str[4:].astype(int)
        gene_2 = edges["gene2"].str[4:].astype(int)
        edges = pd.concat([gene_1, gene_2], axis=1).values.tolist()

        #iterate over the edges end duplicate it because for one edge we need: n1,n2 and n2,n1
        double_edges = []
        for edge in edges:
            double_edges += [ edge, [edge[1], edge[0]]]

        edge_indices = torch.tensor(double_edges)
        edge_indices = edge_indices.t().to(torch.int64).view(2, -1)
        return edge_indices

    def len(self):
        return self.genes.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            graph = torch.load(os.path.join(self.processed_dir, 'graph_test.pt'), weights_only=False)
        else:
            graph = torch.load(os.path.join(self.processed_dir, 'graph.pt'), weights_only=False)

        #return with a given node
        return graph