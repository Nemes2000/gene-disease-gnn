import torch.nn as nn
import torch_geometric.nn as geom_nn

from config import Config

class BasicGNNModel(nn.Module):
    def __init__(self):
        """A basic GNN Model. Layers are builded from GCN layer. 

        """
        super().__init__()
        
        gnn_layer = geom_nn.GCNConv

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
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x