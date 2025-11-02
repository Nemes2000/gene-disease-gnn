import torch.nn as nn
import torch_geometric.nn as geom_nn

from config import Config

class BasicGNNModel(nn.Module):
    def __init__(self):
        """A basic GNN Model. Layers are builded from Config.gnn_layer_type. 

        """
        super().__init__()
        
        if Config.gnn_layer_type == "GCN":
            gnn_layer = geom_nn.GCNConv
        elif Config.gnn_layer_type == "GraphSAGE":
            gnn_layer = geom_nn.SAGEConv
        elif Config.gnn_layer_type == "GAT":
            gnn_layer = lambda in_channels, out_channels: geom_nn.GATConv(in_channels, out_channels, heads=Config.gat_heads, dropout=Config.dropout_rate, concat=False)

        layers = []
        in_channels, hidden_out_channels = Config.in_channels, Config.hidden_channels
        for _ in range(Config.num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=hidden_out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(Config.dropout_rate),
            ]
            in_channels = hidden_out_channels
        
        layers += [gnn_layer(in_channels=in_channels, out_channels=Config.out_channels)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight):
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_weight: List of edge weights

        """
        for layer in self.layers:
            if isinstance(layer, (geom_nn.GCNConv, geom_nn.GATConv)):
                x = layer(x, edge_index, edge_weight=edge_weight)
            elif isinstance(layer, geom_nn.SAGEConv):
                edge_index = edge_index.long()
                x = layer(x, edge_index)
            else:
                x = layer(x)
            
        return x