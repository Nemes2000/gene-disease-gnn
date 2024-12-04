import torch
import tensorflow

import pytorch_lightning as pl
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, ConfusionMatrixDisplay

from models.basic_model import BasicGNNModel
from config import Config


class LightningGNNModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.save_hyperparameters()
        self.loss_function = torch.nn.CrossEntropyLoss()

        if model_name == "basic":
            self.model = BasicGNNModel()
        else:
            self.model = BasicGNNModel()

        # self.cm = ConfusionMatrix(task="binary", num_classes=Config.num_classes)
        # self.aucroc = AUROC(task="binary", num_classes=Config.num_classes)
        # self.f1 = F1Score(task="binary", num_classes=Config.num_classes)
        # self.precision = Precision(task="binary", num_classes=Config.num_classes)
        # self.recall = Recall(task="binary", num_classes=Config.num_classes)

    def forward(self, data, mode="train"):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x_pred = self.model(x, edge_index, edge_weight)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"
        
        loss = self.loss_function(x_pred[mask], data.y[mask])
        acc = (x_pred[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()

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
        x_pred_masked = x_pred[data.test_mask]
        y_masked = data.y[data.test_mask]

        x_pred_binary = tensorflow.cast(x_pred_masked > 0.5, dtype=tensorflow.int32)
        print(x_pred_masked)
        print(x_pred_binary)

        self.log("test_acc", acc)
        self.log('test_loss', loss)
        if len(np.unique(y_masked)) > 1:
            self.log(f"ROC-AUC", roc_auc_score(y_masked, x_pred_binary))

        self.log(f"F1 score", f1_score(y_masked, x_pred_binary))
        self.log(f"Accuracy", accuracy_score(y_masked, x_pred_binary))
        self.log(f"Recall", recall_score(y_masked, x_pred_binary))
        self.log(f"Precision", precision_score(y_masked, x_pred_binary))
        self.log(f"AUPRC", average_precision_score(y_masked, x_pred_binary))
        cm = confusion_matrix(y_masked, x_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig('confusion_matrix.png')

        # self.cm.update(x_masked, y_masked)
        # self.aucroc.update(x_masked, y_masked)
        # self.f1.update(x_masked, y_masked)
        # self.precision.update(x_masked, y_masked)
        # self.recall.update(x_masked, y_masked)
        return loss  

    # def on_test_epoch_end(self) -> None:
    #     self.log('test_auc_roc', self.aucroc.compute(), prog_bar=True, on_epoch=True, batch_size=1)
    #     self.log('test_f1', self.f1.compute(), prog_bar=True, on_epoch=True, batch_size=1)
    #     self.log('test_precision', self.precision.compute(), prog_bar=True, on_epoch=True, batch_size=1)
    #     self.log('test_recall', self.recall.compute(), prog_bar=True, on_epoch=True, batch_size=1)
    #     return super().on_test_epoch_end()

    def configure_optimizers(self):
        return Config.optimizer(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)