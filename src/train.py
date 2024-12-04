import torch_geometric.loader as geom_data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import os

from models.lightning_gnn_model import LightningGNNModel
from config import Config

def callbacks():
    """
    Returns with callbacks for train.
    """

    model_cpkt = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        dirpath='../data/saved_models/wandb',
        filename='best_model')

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        verbose=True,
    )
    return [model_cpkt, early_stopping]

def train_node_classifier(dataset):
    node_data_loader = geom_data.DataLoader(dataset, num_workers=11, persistent_workers=True)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(Config.checkpoint_path, Config.model_name)
    os.makedirs(root_dir, exist_ok=True)

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=callbacks(),
        devices=1,
        max_epochs=Config.epochs,
        logger=pl.loggers.WandbLogger(project=Config.wandb_project_name, log_model="all")
    ) 
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    model = LightningGNNModel(model_name=Config.model_name)

    trainer.fit(model, node_data_loader, node_data_loader)
    model = LightningGNNModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=node_data_loader)
    result = {"test": test_result[0]["test_acc"]}
    return model, result
      
      