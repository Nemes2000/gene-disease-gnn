from sklearn.metrics import accuracy_score
from torch import nn
import torch

from config import Config

class MultiTaskGNNModel(nn.Module):
    def __init__(self, gnn: nn.Module, aux_tasks_num: int):
        super().__init__()
        self.gnn = gnn  # shared encoders
        #TODO: i need an other one for ptetrain???
        self.pr_classifier =  nn.BCEWithLogitsLoss()
        self.aux_classifiers = nn.ModuleList(
            [nn.BCEWithLogitsLoss() for _ in range(aux_tasks_num)]
        )
        
    def forward(self, data, mode = "train", is_pretrain = False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x_pred = self.gnn.forward(x, edge_index, edge_weight)

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
            return self.pr_classifier(x_pred[mask].unsqueeze(1), data.y[mask].unsqueeze(1))
        else:
            pr_mask = mask.clone()
            pr_mask.zero_()
            pr_mask[:, data.pr_disease_idx] = mask[:, data.pr_disease_idx]

            pr_loss = self.pr_classifier(x_pred[mask].unsqueeze(1), data.y[mask].unsqueeze(1))

            aux_losses = []
            for idx, disease_idx in enumerate(data.aux_disease_idxs):
                #TODO: a mask készítés során csak az adott betegség kell vagy a pr betegségre kell osztályozni?
                aux_mask = mask.clone()
                aux_mask.zero_()
                aux_mask[:, disease_idx] = mask[:, disease_idx]

                x_pred_aux = self.aux_classifiers[idx](x_pred[mask].unsqueeze(1), data.y[mask].unsqueeze(1))
                aux_losses.append(x_pred_aux)
           
            return pr_loss, aux_losses
