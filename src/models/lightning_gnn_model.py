import torch
import tensorflow

import pytorch_lightning as pl
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
            self.loss_function = torch.nn.BCEWithLogitsLoss()#FOR every disease --> pos_weight=torch.tensor(Config.pos_class_weight))
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
        
        print(data.y.shape, data.train_mask.shape, data.test_mask.shape)

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

    # def validation_step(self, data):
    #     loss, acc = self.forward(data, mode="val")
    #     self.log("val_acc", acc)
    #     self.log("val_loss", loss)
    #     return loss

    def test_step(self, data):
        global number_for_image
        loss, acc, x_pred = self.forward(data, mode="test")
        x_pred_masked = x_pred[data.test_mask]
        y_masked = data.y[data.test_mask]

        x_pred_binary = torch.where(x_pred_masked > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))

        self.log('test_loss', loss)
        y_masked_cpu = y_masked.cpu()
        if len(np.unique(y_masked_cpu)) > 1:
            self.log("ROC-AUC", roc_auc_score(y_masked_cpu, x_pred_binary))

        self.log("F1 score", f1_score(y_masked_cpu, x_pred_binary))
        self.log("Accuracy", acc)
        self.log("Recall", recall_score(y_masked_cpu, x_pred_binary))
        self.log("Precision", precision_score(y_masked_cpu, x_pred_binary))
        self.log("AUPRC", average_precision_score(y_masked_cpu, x_pred_binary))

        cm = confusion_matrix(y_masked_cpu, x_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig(f'results/matrices/confusion_matrix_{number_for_image}.png')
        number_for_image += 1
        return loss

    # For single disease classication results 
    # def test_step(self,data):
    #     global number_for_image
    #     loss, _, x_pred = self.forward(data, mode="test")

    #     df = pd.DataFrame({"disease_idx": [], "acc": [], "f1": [], "recal": [], "precision": [], "auprc": [], "x_sum": [], "y_sum": []})

    #     for idx, disease_mask in enumerate(data.disease_masks):
    #         x_pred_masked = x_pred[disease_mask].cpu()
    #         y_masked = data.y[disease_mask].cpu()

    #         x_pred_binary = torch.where(x_pred_masked > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
    #         cm = confusion_matrix(y_masked, x_pred_binary)
    #         disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #         disp.plot().figure_.savefig(f'results/matrices/confusion_matrix_{number_for_image}.png')
            
    #         df.loc[len(df)] = {"disease_idx": idx, 
    #                            "acc": accuracy_score(y_masked, x_pred_binary), 
    #                            "f1": f1_score(y_masked, x_pred_binary), 
    #                            "recal": recall_score(y_masked, x_pred_binary), 
    #                            "precision": precision_score(y_masked, x_pred_binary),
    #                            "auprc": average_precision_score(y_masked, x_pred_binary),
    #                            "x_sum": x_pred_binary.sum(),
    #                            "y_sum": y_masked.sum()}
    #         number_for_image += 1
            
    #     df.to_csv("results/disease_classifications.csv", index=False, sep=",")
    #     return loss

    def configure_optimizers(self):
        return Config.optimizer(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
