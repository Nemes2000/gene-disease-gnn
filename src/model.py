import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

# A simple GCN modell for multi-classification
class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super().__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)


    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = F.softmax(self.out(x), dim=1)
        return x