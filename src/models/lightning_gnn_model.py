import torch

import pytorch_lightning as pl

from torchmetrics import ConfusionMatrix, AUROC, F1Score, Precision, Recall

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

        self.cm = ConfusionMatrix(task="binary", num_classes=Config.num_classes)
        self.aucroc = AUROC(task="binary", num_classes=Config.num_classes)
        self.f1 = F1Score(task="binary", num_classes=Config.num_classes)
        self.precision = Precision(task="binary", num_classes=Config.num_classes)
        self.recall = Recall(task="binary", num_classes=Config.num_classes)

    def forward(self, data, mode="train"):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        new_x = self.model(x, edge_index, edge_weight)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.loss_function(new_x[mask], data.y[mask])
        acc = (new_x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()

        if mode == "test":
            return loss, acc, new_x
        return loss, acc

    def training_step(self, data):
        loss, acc = self.forward(data, mode="train")
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, batch_size=1)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data):
        loss, acc = self.forward(data, mode="val")
        self.log("val_acc", acc, batch_size=1)
        self.log("val_loss", loss, batch_size=1)

    def test_step(self, data):
        loss, acc, x = self.forward(data, mode="test")
        x_masked = x[data.test_mask]
        y_masked = data.y[data.test_mask]

        self.log("test_acc", acc, batch_size=1)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, batch_size=1)
        self.cm.update(x_masked, y_masked)
        self.aucroc.update(x_masked, y_masked)
        self.f1.update(x_masked, y_masked)
        self.precision.update(x_masked, y_masked)
        self.recall.update(x_masked, y_masked)
        return loss

    def on_test_epoch_end(self) -> None:
        self.cm.plot()
        self.log('test_auc_roc', self.aucroc.compute(), prog_bar=True, on_epoch=True, batch_size=1)
        self.log('test_f1', self.f1.compute(), prog_bar=True, on_epoch=True, batch_size=1)
        self.log('test_precision', self.precision.compute(), prog_bar=True, on_epoch=True, batch_size=1)
        self.log('test_recall', self.recall.compute(), prog_bar=True, on_epoch=True, batch_size=1)
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        return Config.optimizer(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)