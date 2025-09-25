from sklearn.metrics import accuracy_score
from torch import nn
import copy

from config import Config

class MultiTaskGNNModel(nn.Module):
    def __init__(self, gnn: nn.Module, aux_tasks_num: int):
        super().__init__()
        self.gnn = gnn  # shared encoders

        #TODO: create complex layers for private layers
        private_layer = nn.Sequential(
            nn.Linear(Config.out_channels, 1000),
            nn.ReLU(inplace=True), 
            nn.Dropout(Config.dropout_rate), 
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True), 
            nn.Dropout(Config.dropout_rate),
            nn.Linear(500, Config.out_channels),
            )

        self.pr_layer = private_layer
        self.aux_layers = nn.ModuleList([copy.deepcopy(private_layer) for _ in range(aux_tasks_num)])
        self.classifier =  nn.BCEWithLogitsLoss()
        
    def forward(self, data, mode = "train", is_pretrain = False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        embeding = self.gnn.forward(x, edge_index, edge_weight)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.test_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        if is_pretrain:
            return self.classifier(embeding[mask].unsqueeze(1), data.y[mask].unsqueeze(1))
        else:
            pr_mask = mask.clone()
            pr_mask.zero_()
            pr_mask[:, Config.pr_disease_idx] = mask[:, Config.pr_disease_idx]

            pr_embeding = self.pr_layer(embeding)

            pr_loss = self.classifier(pr_embeding[pr_mask].unsqueeze(1), data.y[pr_mask].unsqueeze(1))

            aux_losses = []
            for idx, disease_idx in enumerate(Config.aux_disease_idxs):
                aux_mask = mask.clone()
                aux_mask.zero_()
                aux_mask[:, disease_idx] = mask[:, disease_idx]

                aux_embeding = self.aux_layers[idx](embeding)

                x_pred_aux = self.classifier(aux_embeding[aux_mask].unsqueeze(1), data.y[aux_mask].unsqueeze(1))
                aux_losses.append(x_pred_aux)

            if mode == "test":
                return pr_loss, embeding
           
            return pr_loss, aux_losses
