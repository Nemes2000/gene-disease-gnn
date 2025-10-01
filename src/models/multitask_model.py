from sklearn.metrics import accuracy_score
from torch import nn
import copy
import torch

from config import Config

class MultiTaskGNNModel(nn.Module):
    def __init__(self, gnn: nn.Module, aux_tasks_num: int):
        super().__init__()
        self.gnn = gnn  # shared encoders

        #TODO: feature importance score a sulyozó modellre
        
        private_layer = nn.Sequential(
            nn.Linear(Config.out_channels, 1),
            # nn.ReLU(inplace=True), 
            # nn.Dropout(Config.dropout_rate), 
            # nn.Linear(Config.mt_hidden_1, Config.mt_hidden_2),
            # nn.ReLU(inplace=True),
            # nn.Dropout(Config.dropout_rate),
            # nn.Linear(Config.mt_hidden_2, 1),
            ) # shape: [gene number, 1]

        self.pr_layer = private_layer
        self.aux_layers = nn.ModuleList([copy.deepcopy(private_layer) for _ in range(aux_tasks_num)])
        self.pr_classifier = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(Config.pr_pos_class_weight), reduction="none") # shape: [gene number, 1]
        self.aux_classifiers = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w), reduction="none") for w in Config.aux_pos_class_weights]
        
    def forward(self, data, mode = "train", is_pretrain = False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        embeding = self.gnn.forward(x, edge_index, edge_weight)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        if is_pretrain:
            return self.pr_classifier(embeding[mask].unsqueeze(1), data.y[mask].unsqueeze(1))
        else:
            #(gene num, 1)
            pr_embeding = self.pr_layer(embeding)
            pr_loss = self.pr_classifier(pr_embeding, data.y[:, Config.pr_disease_idx].unsqueeze(1))
            
            if mode == "test":
                return pr_loss, pr_embeding

            aux_losses = []
            for disease_idx, classifier, pr_layer in zip(Config.aux_disease_idxs, self.aux_classifiers, self.aux_layers):
                aux_embeding = pr_layer(embeding)
                x_pred_aux = classifier(aux_embeding, data.y[:, disease_idx].unsqueeze(1))
                aux_losses.append(x_pred_aux)
           
            return pr_loss, aux_losses #shape [gene num, 1]
