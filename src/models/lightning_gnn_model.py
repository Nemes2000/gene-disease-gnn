import torch

import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, ConfusionMatrixDisplay

from models.basic_model import BasicGNNModel
from config import Config, ModelTypes

number_for_image = 0


class LightningGNNModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.save_hyperparameters()

        if model_name == ModelTypes.BASIC:
            self.loss_function = torch.nn.BCEWithLogitsLoss()
            self.model = BasicGNNModel()
        elif model_name == ModelTypes.CLS_WEIGHT:
            pos_weight=torch.tensor(Config.pos_class_weight)
            self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)#FOR every disease --> 
            self.model = BasicGNNModel()

    def forward(self, data, mode="train"):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x_pred = self.model.forward(x, edge_index, edge_weight)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.test_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"
        
        if Config.disease_idx:
            mask_t = mask.clone()
            mask_t.zero_()
            mask_t[:, Config.disease_idx] = mask[:, Config.disease_idx]
            mask = mask_t

        # After masking out the given percentage of the matrix i cant recreate the original one with same shape
        # disease_num = data.y.shape[1]
        # x_pred[mask].reshape(-1, disease_num)

        loss = self.loss_function(x_pred[mask].unsqueeze(1), data.y[mask].unsqueeze(1))

        x_pred_binary = torch.where(x_pred[mask] > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
        y_masked = data.y[mask]
        acc = accuracy_score(y_masked, x_pred_binary)

        if mode == "test":
            return loss, acc, x_pred
        return loss, acc

    def training_step(self, data):
        loss, acc = self.forward(data, mode="train")
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, data):
        loss, acc = self.forward(data, mode="val")
        self.log("val_acc", acc)
        self.log("val_loss", loss)
        return loss

    def test_step(self, data):
        loss, acc, x_pred = self.forward(data, mode="test")

        if Config.disease_idx:
            x_pred_masked = x_pred[:, Config.disease_idx]
            y_masked = data.y[:, Config.disease_idx]
        else:
            x_pred_masked = x_pred[data.test_mask]
            y_masked = data.y[data.test_mask]

        x_pred_binary = torch.where(x_pred_masked > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))

        self.log('test_loss', loss)
        if len(np.unique(y_masked)) > 1:
            self.log("ROC-AUC", roc_auc_score(y_masked, x_pred_binary))

        self.log("F1 score", f1_score(y_masked, x_pred_binary))
        self.log("Accuracy", acc)
        self.log("Recall", recall_score(y_masked, x_pred_binary))
        self.log("Precision", precision_score(y_masked, x_pred_binary))
        self.log("AUPRC", average_precision_score(y_masked, x_pred_binary))

        cm = confusion_matrix(y_masked, x_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig(f'results/matrices/confusion_matrix_{Config.disease_idx}.png')
        with open(f'results/matrices/confusion_matrix_{Config.disease_idx}.txt', "w") as file:
            file.write(f"""acc: {accuracy_score(y_masked, x_pred_binary)}, 
f1: {f1_score(y_masked, x_pred_binary)}, 
recal: {recall_score(y_masked, x_pred_binary)}, 
precision: {precision_score(y_masked, x_pred_binary)},
auc: {roc_auc_score(y_masked, x_pred_binary)},
auprc: {average_precision_score(y_masked, x_pred_binary)},
cm: {" ".join(map(str, cm.flatten()))},
x_sum: {x_pred_binary.sum().item()},
y_sum: {y_masked.sum().item()}""")

        return loss

    # For single disease classication results 
    # def test_step(self,data):
    #     loss, _, x_pred = self.forward(data, mode="test")

    #     df = pd.DataFrame({"disease_idx": [], "acc": [], "f1": [], "recal": [], "precision": [], "auc":[],"auprc": [], "cm": [], "x_sum": [], "y_sum": []})
    #     print("Creating ped disease statisctic...")
    #     for idx in range(data.y.shape[1]):
    #         x_pred_masked = x_pred[:, idx]
    #         y_masked = data.y[:, idx]

    #         x_pred_binary = torch.where(x_pred_masked > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
    #         cm = confusion_matrix(y_masked, x_pred_binary)

    #         auc = 0
    #         if len(np.unique(y_masked)) > 1:
    #             auc = roc_auc_score(y_masked, x_pred_binary)
            
    #         df.loc[len(df)] = {"disease_idx": idx, 
    #                            "acc": accuracy_score(y_masked, x_pred_binary), 
    #                            "f1": f1_score(y_masked, x_pred_binary), 
    #                            "recal": recall_score(y_masked, x_pred_binary), 
    #                            "precision": precision_score(y_masked, x_pred_binary),
    #                            "auc": auc,
    #                            "auprc": average_precision_score(y_masked, x_pred_binary),
    #                            "cm": " ".join(map(str, cm.flatten())),
    #                            "x_sum": x_pred_binary.sum().item(),
    #                            "y_sum": y_masked.sum().item()}
            
    #     df.to_csv("results/disease_classifications.csv", index=False, sep=",")
    #     print("Creating ped disease statisctic. DONE")
    #     return loss

    def configure_optimizers(self):
        return Config.optimizer(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
