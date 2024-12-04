from torch_geometric.data import Dataset, Data
import torch

import pandas as pd
import numpy as np

import os
from tqdm import tqdm

from config import Config
from mappers.idmapper import IdMapper

class GeneDataset(Dataset):
    def __init__(self, root, filenames, test_size, val_size, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.files_parent_dir = Config.raw_data_path
        file_0_path = os.path.join(os.path.dirname(__file__), self.files_parent_dir+filenames[0])
        file_1_path = os.path.join(os.path.dirname(__file__), self.files_parent_dir+filenames[1])
        file_2_path = os.path.join(os.path.dirname(__file__), self.files_parent_dir+filenames[2])

        # Normalize the path (optional, for cross-platform compatibility)
        file_0_path = os.path.normpath(file_0_path)
        file_1_path = os.path.normpath(file_1_path)
        file_2_path = os.path.normpath(file_2_path)

        self.test = test
        self.test_size = test_size
        self.val_size = val_size
        self.filenames = [file_0_path, file_1_path, file_2_path]
        self.mapper = IdMapper(file_0_path, file_2_path)
        super(GeneDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_dir(self):
        return os.path.join(os.path.dirname(__file__), Config.processed_data_dir)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filenames

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        file_0_path = os.path.join(os.path.dirname(__file__), self.files_parent_dir+self.filenames[0])
        file_1_path = os.path.join(os.path.dirname(__file__), self.files_parent_dir+self.filenames[1])
        file_2_path = os.path.join(os.path.dirname(__file__), self.files_parent_dir+self.filenames[2])
        return [file_0_path, file_1_path, file_2_path]

    def download(self):
        pass

    def process(self):
        self.genes_features = pd.read_csv(self.raw_paths[0], sep="\t")
        self.edges_features = pd.read_csv(self.raw_paths[1], sep="\t")
        self.disiese_gene_matrix = pd.read_csv(self.raw_paths[2], sep="\t")

        self.genes = self.genes_features["genes"].sort_values().unique()
        self.diseases = self.disiese_gene_matrix["diseaseId"].sort_values().unique()

        node_feats = self._get_node_features(self.genes_features)
        edge_feats = self._get_edge_features(self.edges_features)
        edge_index = self._get_adjacency_info(self.edges_features)

        y = self._create_mask_matrix(self.disiese_gene_matrix.copy()).to(torch.float32)
        train_mask, validation_mask, test_mask = self._get_train_val_test_mask(self.disiese_gene_matrix.copy())

        data = Data(x=node_feats,
                    edge_index=edge_index,
                    edge_weight=edge_feats,
                    test_mask=test_mask, val_mask=validation_mask, train_mask=train_mask, y=y)

        torch.save(data, os.path.join(self.processed_dir, 'graph.pt'))


    def _get_train_val_test_mask(self, disiese_gene_matrix):
        """
        i need too create matrices shape like disgenet
        and in this matrix i pick random points which are gonna be the train mask, validation mask and test mask

        in the train dataset i need to pick 80% from disgenet, equaly 0s and 1s in a column
        in the validation dataset i need to pick 10% from disgenet, equaly 0s and 1s in a column
        """

        train, validation, test = self._split_labels_to_train_val_test(disiese_gene_matrix)
        disgenet_inverse = self._get_disgenet_inverse(disiese_gene_matrix)
        train_n, validation_n, test_n = self._split_labels_to_train_val_test(disgenet_inverse)
        train_r = pd.concat([train, train_n], ignore_index=True)
        validation_r = pd.concat([validation, validation_n], ignore_index=True)
        test_r = pd.concat([test, test_n], ignore_index=True)

        train_mask = self._create_mask_matrix(train_r)
        validation_mask = self._create_mask_matrix(validation_r)
        test_mask = self._create_mask_matrix(test_r)

        return train_mask, validation_mask, test_mask

    def _split_labels_to_train_val_test(self, disgenet: pd.DataFrame):
        #Split the positive targets to equal partitions by disease
        disgenet_grouped = disgenet.groupby(by="diseaseId", group_keys=False)
        test_validation = disgenet_grouped.apply(lambda x: x.sample(frac=Config.train_test_split, random_state=1))
        train = disgenet.drop(test_validation.index)
        test_validation_grouped = test_validation.groupby(by="diseaseId", group_keys=False)

        #Group by is needed before sample function call!!!
        test = test_validation_grouped.apply(lambda x: x.sample(frac=Config.test_val_split, random_state=1))
        drop_indices = pd.concat([train, test]).index
        validation = disgenet.drop(drop_indices)
        return train, validation, test


    def _get_disgenet_inverse(self, disgenet):
        genes_frame = pd.DataFrame(list(self.genes), columns=["geneId"])
        diseases_frame = pd.DataFrame(self.diseases, columns=["diseaseId"])
        gene_disease_descartes_product = genes_frame.merge(diseases_frame, how="cross")
        disgenet_inverse = gene_disease_descartes_product.merge(disgenet, on=['geneId', 'diseaseId'], how='left', indicator=True)
        return disgenet_inverse[disgenet_inverse['_merge'] == 'left_only'].drop(columns='_merge')


    def _create_mask_matrix(self, dataframe):
        dataframe_for_matrix = pd.DataFrame(np.zeros((len(self.genes), len(self.diseases)),))
        gene_id_to_idx = self.mapper.genes_id_to_idx_map()
        disease_id_to_idx = self.mapper.diseases_id_to_idx_map()

        dataframe["geneId"] = dataframe["geneId"].map(gene_id_to_idx)
        dataframe["diseaseId"] = dataframe["diseaseId"].map(disease_id_to_idx)
        tuples_array = [row for row in dataframe.itertuples(index=False, name=None)]
        for row, col in tqdm(tuples_array):
            dataframe_for_matrix.loc[row, col] = 1

        return torch.tensor(dataframe_for_matrix.to_numpy(), dtype=torch.bool)

    def _get_node_features(self, genes):
        gene_id_to_idx = self.mapper.genes_id_to_idx_map()
        genes["genes"] = self.genes_features["genes"].map(gene_id_to_idx)
        all_node_feats = genes.values.tolist()
        all_node_feats = np.asarray(all_node_feats)

        return torch.tensor(all_node_feats, dtype=torch.float32)

    def _get_edge_features(self, edges):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        duplicated_edges = edges.loc[edges.index.repeat(2)].reset_index(drop=True)
        all_edge_feats = duplicated_edges["combined_score"].tolist()
        return torch.tensor(all_edge_feats, dtype=torch.float32)


    def _get_adjacency_info(self, edges):
        """
        We want to be sure that the order of the indices
        matches the order of the edge features
        """
        gene_id_to_idx = self.mapper.genes_id_to_idx_map()

        edge_indices = []
        gene_1 = edges["gene1"].map(gene_id_to_idx)
        gene_2 = edges["gene2"].map(gene_id_to_idx)
        edges = pd.concat([gene_1, gene_2], axis=1).values.tolist()

        #iterate over the edges end duplicate it because for one edge we need: n1,n2 and n2,n1
        double_edges = []
        for edge in edges:
            double_edges += [ edge, [edge[1], edge[0]]]

        edge_indices = torch.tensor(double_edges)
        edge_indices = edge_indices.t().to(torch.int32).view(2, -1)
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

        return graph

    def __getitem__(self, _):
        return self.get(0)